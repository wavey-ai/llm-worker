//! Local LLM inference engine.
//!
//! The engine owns the model and its context on a dedicated OS thread.
//! `llama.cpp`'s `decode` is synchronous, blocks for the duration of a forward
//! pass, and needs `&mut` access to the context, so it cannot run on an async
//! runtime's worker threads or be shared between tasks. Callers submit work
//! over a channel and receive tokens back over another; nothing in this module
//! knows about HTTP.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::sync::{Arc, OnceLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use anyhow::{Context, Result, anyhow};
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::{LogOptions, ggml_time_us, send_logs_to_tracing};
use tokio::sync::mpsc;
use tracing::{debug, info, info_span, trace, warn};

/// Where to get the model from.
#[derive(Debug, Clone)]
pub enum ModelSource {
    /// A GGUF file already on disk.
    Local(PathBuf),
    /// A file in a Hugging Face repo, resolved through the shared HF cache.
    HuggingFace { repo: String, file: String },
}

impl ModelSource {
    fn resolve(&self) -> Result<PathBuf> {
        match self {
            Self::Local(path) => {
                // llama.cpp asserts rather than returning an error for a
                // missing file, and an assert on the engine thread reaches the
                // caller as "the thread exited" with the real cause on stderr.
                anyhow::ensure!(path.is_file(), "no model file at {}", path.display());
                debug!(path = %path.display(), "using local model");
                Ok(path.clone())
            }
            Self::HuggingFace { repo, file } => {
                let span = info_span!("hf_fetch", %repo, %file);
                let _guard = span.enter();
                let path = hf_hub::api::sync::ApiBuilder::new()
                    .with_progress(true)
                    .build()
                    .context("unable to create the Hugging Face client")?
                    .model(repo.clone())
                    .get(file)
                    .with_context(|| format!("unable to fetch {repo}/{file}"))?;
                debug!(path = %path.display(), "resolved from cache");
                Ok(path)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub model: ModelSource,
    /// Total context across all sequences.
    pub n_ctx: u32,
    /// Layers to offload. Clamped to the model's real layer count.
    pub n_gpu_layers: u32,
    /// How many generations run at once. One until the multi-slot loop
    /// lands; then it is llama.cpp's `n_seq_max`. This is the number a
    /// scheduler sizes its queue against, so it must mean concurrency and
    /// not queue length.
    pub max_inflight: usize,
    /// How many requests may wait for a slot before `submit` reports the
    /// engine full. Backpressure of last resort: a caller with a queue of its
    /// own should be reading `capacity()` long before this bites.
    pub queue_depth: usize,
    /// Route llama.cpp's own logs into `tracing` under the `llama-cpp-2` target.
    pub native_logs: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model: ModelSource::HuggingFace {
                repo: "unsloth/Qwen3.5-0.8B-GGUF".to_string(),
                file: "Qwen3.5-0.8B-Q4_K_M.gguf".to_string(),
            },
            n_ctx: 8192,
            n_gpu_layers: 99,
            max_inflight: 1,
            queue_depth: 32,
            native_logs: true,
        }
    }
}

/// One turn of a conversation. Roles are passed to the model's own chat
/// template verbatim, so what counts as valid is the template's business,
/// not ours.
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".to_string(), content: content.into() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".to_string(), content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".to_string(), content: content.into() }
    }
}

/// One unit of work. The engine applies the model's own chat template, so
/// these are the conversation's turns, not a formatted string.
#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub messages: Vec<Message>,
    pub max_tokens: i32,
    /// Zero is greedy and deterministic.
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
    /// Let the model emit its reasoning block.
    pub think: bool,
}

impl Default for GenerateRequest {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            max_tokens: 128,
            temperature: 0.0,
            top_p: 0.9,
            seed: 42,
            think: false,
        }
    }
}

impl GenerateRequest {
    /// The single-turn case, which is most of them.
    pub fn user(prompt: impl Into<String>) -> Self {
        Self { messages: vec![Message::user(prompt)], ..Self::default() }
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stop {
    /// The model emitted an end-of-generation token.
    EndOfGeneration,
    /// `max_tokens` was reached.
    Limit,
    /// The context window filled. Generation cannot continue without
    /// discarding history, which is a decision for the caller, not the engine.
    ContextFull,
    /// The caller cancelled, or dropped the `Generation`.
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum Event {
    /// A decoded piece of text. Never a partial UTF-8 sequence.
    Token(String),
    Done {
        stop: Stop,
        /// Tokens in the templated prompt, counted at prefill.
        prompt_tokens: u32,
        /// Tokens generated.
        tokens: u32,
        elapsed_ms: u64,
    },
    Failed(String),
}

#[derive(Debug, Clone, Copy)]
pub struct Capacity {
    pub inflight: usize,
    pub max_inflight: usize,
    pub available_slots: usize,
}

#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub path: PathBuf,
    pub n_params: u64,
    pub n_layer: u32,
    pub n_ctx: u32,
}

struct Job {
    request: GenerateRequest,
    events: mpsc::Sender<Event>,
    cancel: Arc<AtomicBool>,
}

/// Handle to the engine thread. Cheap to clone; the thread is shut down and
/// joined when the last clone drops.
#[derive(Clone)]
pub struct Engine {
    inner: Arc<Inner>,
}

struct Inner {
    jobs: Option<mpsc::Sender<Job>>,
    thread: Option<std::thread::JoinHandle<()>>,
    inflight: Arc<AtomicUsize>,
    max_inflight: usize,
    info: ModelInfo,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Closing the queue ends the engine loop, which drops the model and
        // then the backend. Joining before we return guarantees that GPU
        // teardown finishes rather than racing process exit.
        drop(self.jobs.take());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Engine {
    /// Load the model and start the engine thread. Blocks until the model is
    /// resident, so a caller can register for work only once it can serve it.
    pub fn spawn(config: EngineConfig) -> Result<Self> {
        let max_inflight = config.max_inflight.max(1);
        // A queue shorter than the slot count would refuse work the engine
        // has room for, so the queue is at least as deep as the engine is wide.
        let queue_depth = config.queue_depth.max(max_inflight);
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job>(queue_depth);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<ModelInfo>>();
        let inflight = Arc::new(AtomicUsize::new(0));

        let thread_inflight = inflight.clone();
        let thread = std::thread::Builder::new()
            .name("llm-engine".to_string())
            .spawn(move || engine_thread(config, jobs_rx, ready_tx, thread_inflight))
            .context("failed to start the engine thread")?;

        let info = ready_rx
            .recv()
            .map_err(|_| anyhow!("engine thread exited during startup"))??;

        Ok(Self {
            inner: Arc::new(Inner {
                jobs: Some(jobs_tx),
                thread: Some(thread),
                inflight,
                max_inflight,
                info,
            }),
        })
    }

    pub fn model_info(&self) -> &ModelInfo {
        &self.inner.info
    }

    /// Slot occupancy, shaped for a scheduler's heartbeat. `inflight` counts
    /// everything submitted and not yet finished, so a request waiting behind
    /// another shows up as occupancy rather than as free capacity.
    pub fn capacity(&self) -> Capacity {
        let inflight = self.inner.inflight.load(Ordering::Acquire);
        Capacity {
            inflight,
            max_inflight: self.inner.max_inflight,
            available_slots: self.inner.max_inflight.saturating_sub(inflight),
        }
    }

    /// Queue a request. Returns immediately; tokens arrive on the `Generation`.
    ///
    /// Errors when the queue is full, which is the engine's own backpressure
    /// and is deliberately independent of any transport's.
    pub fn submit(&self, request: GenerateRequest) -> Result<Generation> {
        let (events_tx, events_rx) = mpsc::channel::<Event>(256);
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job { request, events: events_tx, cancel: cancel.clone() };

        let jobs = self
            .inner
            .jobs
            .as_ref()
            .ok_or_else(|| anyhow!("engine has shut down"))?;
        jobs.try_send(job).map_err(|err| match err {
            mpsc::error::TrySendError::Full(_) => anyhow!("engine queue is full"),
            mpsc::error::TrySendError::Closed(_) => anyhow!("engine has shut down"),
        })?;

        self.inner.inflight.fetch_add(1, Ordering::AcqRel);
        Ok(Generation { events: events_rx, cancel })
    }
}

/// A running generation. Dropping it cancels the work.
pub struct Generation {
    events: mpsc::Receiver<Event>,
    cancel: Arc<AtomicBool>,
}

impl Generation {
    /// Next event, or `None` once the engine is finished with this request.
    pub async fn recv(&mut self) -> Option<Event> {
        self.events.recv().await
    }

    /// Stop generating. Takes effect at the next token boundary; an in-flight
    /// forward pass is not interrupted.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Acquire)
    }
}

impl Drop for Generation {
    fn drop(&mut self) {
        // A dropped receiver means nobody is listening: a disconnected client
        // must not keep occupying a slot.
        self.cancel();
    }
}

fn engine_thread(
    config: EngineConfig,
    mut jobs: mpsc::Receiver<Job>,
    ready: std::sync::mpsc::Sender<Result<ModelInfo>>,
    inflight: Arc<AtomicUsize>,
) {
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(config.native_logs));

    let started = match Runtime::load(&config) {
        Ok(runtime) => runtime,
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };
    let mut runtime = started;
    if ready.send(Ok(runtime.info.clone())).is_err() {
        return; // Caller gave up while we were loading.
    }

    while let Some(job) = jobs.blocking_recv() {
        let outcome = runtime.run(&job);

        // Release the slot *before* announcing completion, so a caller that
        // reads `capacity()` on seeing the terminal event never sees a slot
        // that is finished but still counted.
        inflight.fetch_sub(1, Ordering::AcqRel);

        let event = match outcome {
            Ok(Outcome { stop, prompt_tokens, tokens, elapsed_ms }) => Event::Done {
                stop,
                prompt_tokens,
                tokens,
                elapsed_ms,
            },
            Err(err) => {
                warn!(error = %err, "generation failed");
                Event::Failed(err.to_string())
            }
        };
        let _ = job.events.blocking_send(event);
    }
}

/// llama.cpp's backend is process-global: `LlamaBackend::init` fails with
/// `BackendAlreadyInitialized` on a second call, so it cannot be per-engine.
/// Holding it in a `OnceLock` also means the Metal device outlives every model
/// and context, which is what its teardown assertions require.
fn shared_backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| LlamaBackend::init().map_err(|err| err.to_string()))
        .as_ref()
        .map_err(|err| anyhow!("failed to initialise the llama backend: {err}"))
}

/// What a finished job produced. The terminal event is sent by the engine loop
/// rather than here, so slot accounting settles first.
struct Outcome {
    stop: Stop,
    prompt_tokens: u32,
    tokens: u32,
    elapsed_ms: u64,
}

struct Runtime {
    model: LlamaModel,
    backend: &'static LlamaBackend,
    info: ModelInfo,
    n_ctx: u32,
}

impl Runtime {
    fn load(config: &EngineConfig) -> Result<Self> {
        let path = config.model.resolve()?;
        let span = info_span!("load", path = %path.display());
        let _guard = span.enter();
        let started = ggml_time_us();

        let backend = shared_backend()?;
        let params = pin!(LlamaModelParams::default().with_n_gpu_layers(config.n_gpu_layers));
        let model = LlamaModel::load_from_file(backend, &path, &params)
            .with_context(|| format!("failed to load {}", path.display()))?;

        info!(
            elapsed_ms = (ggml_time_us() - started) / 1_000,
            params = model.n_params(),
            n_layer = model.n_layer(),
            n_head_kv = model.n_head_kv(),
            n_vocab = model.n_vocab(),
            "model loaded"
        );

        let info = ModelInfo {
            path,
            n_params: model.n_params(),
            n_layer: model.n_layer(),
            n_ctx: config.n_ctx,
        };
        Ok(Self { model, backend, info, n_ctx: config.n_ctx })
    }

    fn run(&mut self, job: &Job) -> Result<Outcome> {
        if job.cancel.load(Ordering::Acquire) {
            // Cancelled while queued: never touch the GPU for it.
            return Ok(Outcome {
                stop: Stop::Cancelled,
                prompt_tokens: 0,
                tokens: 0,
                elapsed_ms: 0,
            });
        }

        // A fresh context per request keeps this single-slot version honest:
        // no KV state survives between requests. Multi-slot will hold one
        // context and partition it by sequence id instead.
        let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(self.n_ctx));
        let mut ctx = self
            .model
            .new_context(self.backend, ctx_params)
            .context("failed to create the llama context")?;

        let prompt = self.build_prompt(&job.request)?;
        let tokens = self
            .model
            .str_to_token(&prompt, AddBos::Always)
            .context("failed to tokenize the prompt")?;

        // A prompt that does not fit cannot be decoded at all, and the error
        // llama.cpp gives for it says nothing about why.
        anyhow::ensure!(
            (tokens.len() as u32) < self.n_ctx,
            "prompt is {} tokens and the context is {}",
            tokens.len(),
            self.n_ctx
        );

        let prefill = info_span!("prefill", prompt_tokens = tokens.len()).entered();
        let started = ggml_time_us();
        let mut batch = LlamaBatch::new(tokens.len().max(512), 1);
        let last = tokens.len() as i32 - 1;
        for (i, token) in (0i32..).zip(tokens.iter().copied()) {
            batch.add(token, i, &[0], i == last)?;
        }
        ctx.decode(&mut batch).context("prompt decode failed")?;
        let prefill_s = (ggml_time_us() - started) as f64 / 1e6;
        info!(
            elapsed_ms = (prefill_s * 1e3) as u64,
            tok_per_sec = tokens.len() as f64 / prefill_s,
            "prompt ingested"
        );
        drop(prefill);

        let mut sampler = if job.request.temperature > 0.0 {
            LlamaSampler::chain_simple([
                LlamaSampler::temp(job.request.temperature),
                LlamaSampler::top_p(job.request.top_p, 1),
                LlamaSampler::dist(job.request.seed),
            ])
        } else {
            LlamaSampler::chain_simple([LlamaSampler::greedy()])
        };

        // A token is a byte sequence, not a character, so a multi-byte
        // character can straddle two tokens. The incremental decoder holds the
        // partial bytes until they form one.
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut n_cur = batch.n_tokens();
        // Decoding past the context window fails inside llama.cpp with a
        // memory-slot error, mid-sentence. Stop at the edge and say so.
        let room = self.n_ctx as i32 - n_cur;
        let n_limit = n_cur + job.request.max_tokens.min(room);
        let context_bound = job.request.max_tokens > room;
        let mut n_decoded: u32 = 0;

        let generate = info_span!("generate", max_tokens = job.request.max_tokens).entered();
        let started = ggml_time_us();
        let mut stop = if context_bound { Stop::ContextFull } else { Stop::Limit };

        while n_cur < n_limit {
            if job.cancel.load(Ordering::Acquire) {
                stop = Stop::Cancelled;
                break;
            }

            let token = sampler.sample(&ctx, batch.n_tokens() - 1);
            sampler.accept(token);

            if self.model.is_eog_token(token) {
                stop = Stop::EndOfGeneration;
                break;
            }

            let piece = self.model.token_to_piece(token, &mut decoder, false, None)?;
            trace!(token = token.0, %piece, "token");
            if job.events.blocking_send(Event::Token(piece)).is_err() {
                // Receiver gone. Drop sets the cancel flag too, but a closed
                // channel is the earlier and more direct signal.
                stop = Stop::Cancelled;
                break;
            }

            batch.clear();
            batch.add(token, n_cur, &[0], true)?;
            n_cur += 1;
            n_decoded += 1;

            ctx.decode(&mut batch).context("decode failed")?;
        }

        let elapsed = (ggml_time_us() - started) as f64 / 1e6;
        info!(
            tokens = n_decoded,
            elapsed_ms = (elapsed * 1e3) as u64,
            tok_per_sec = f64::from(n_decoded) / elapsed.max(f64::EPSILON),
            ?stop,
            "generation complete"
        );
        drop(generate);

        // The context is dropped here, releasing its KV cache. For a cancelled
        // request that is the whole of the cleanup; multi-slot will call
        // kv_cache_seq_rm on the slot instead.
        Ok(Outcome {
            stop,
            prompt_tokens: tokens.len() as u32,
            tokens: n_decoded,
            elapsed_ms: (elapsed * 1e3) as u64,
        })
    }

    fn build_prompt(&self, request: &GenerateRequest) -> Result<String> {
        anyhow::ensure!(!request.messages.is_empty(), "request has no messages");
        let template = self
            .model
            .chat_template(None)
            .context("model has no chat template")?;
        let messages = request
            .messages
            .iter()
            .map(|message| {
                LlamaChatMessage::new(message.role.clone(), message.content.clone())
                    .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        let mut prompt = self.model.apply_chat_template(&template, &messages, true)?;

        // apply_chat_template has no route to the template's `enable_thinking`
        // argument, so close the reasoning block by hand to get non-thinking mode.
        if !request.think {
            prompt.push_str("<think>\n\n</think>\n\n");
        }
        trace!(%prompt, "templated prompt");
        Ok(prompt)
    }
}
