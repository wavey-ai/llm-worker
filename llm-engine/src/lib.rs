//! Local LLM inference engine.
//!
//! The engine owns the model and one shared context on a dedicated OS thread,
//! decoding every active request together in a single forward pass.
//! `llama.cpp`'s `decode` is synchronous, blocks for the duration of a forward
//! pass, and needs `&mut` access to the context, so it cannot run on an async
//! runtime's worker threads or be shared between tasks. Callers submit work
//! over a channel and receive tokens back over another; nothing in this module
//! knows about HTTP.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
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
    jobs: mpsc::Receiver<Job>,
    ready: std::sync::mpsc::Sender<Result<ModelInfo>>,
    inflight: Arc<AtomicUsize>,
) {
    send_logs_to_tracing(LogOptions::default().with_logs_enabled(config.native_logs));

    let runtime = match Runtime::load(&config) {
        Ok(runtime) => runtime,
        Err(err) => {
            let _ = ready.send(Err(err));
            return;
        }
    };

    // A context borrows the model it came from, so the two cannot live side by
    // side in one struct. It lives inside `serve` instead, for exactly as long
    // as the engine does, which also keeps the teardown order right: context,
    // then model, then the backend that outlives both.
    runtime.serve(jobs, ready, inflight);
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

/// How long a slot may hold its sequence while the caller refuses to read
/// before the engine takes the slot back.
const STALL_TIMEOUT: Duration = Duration::from_secs(30);

/// Idle wait when there is nothing to decode but something still to deliver.
const IDLE_TICK: Duration = Duration::from_millis(1);

/// One request, occupying one sequence of the shared context.
struct Slot {
    job: Job,
    sampler: LlamaSampler,
    /// A token is a byte sequence, not a character, so a multi-byte character
    /// can straddle two tokens. This holds the partial bytes until they form
    /// one, and so is per slot rather than per engine.
    decoder: encoding_rs::Decoder,
    seq_id: u32,
    /// Position the next token will occupy.
    n_cur: i32,
    /// The token to feed at `n_cur`. Already sent to the caller.
    next_input: LlamaToken,
    prompt_tokens: u32,
    n_decoded: u32,
    /// Tokens this slot may still generate.
    budget: u32,
    /// Whether that budget came from the context rather than the request.
    context_bound: bool,
    started_us: i64,
    /// A piece the caller has not taken yet. While this is set the slot stays
    /// out of the batch: one slow reader must not hold up the others.
    pending: Option<Event>,
    stalled_since: Option<Instant>,
}

impl Slot {
    /// Try to hand the caller its last piece. A slot with nothing pending is
    /// ready to decode again.
    fn flush(&mut self) -> Flush {
        let Some(event) = self.pending.take() else {
            return Flush::Ready;
        };
        match self.job.events.try_send(event) {
            Ok(()) => {
                self.stalled_since = None;
                Flush::Ready
            }
            Err(mpsc::error::TrySendError::Full(event)) => {
                self.pending = Some(event);
                let since = *self.stalled_since.get_or_insert_with(Instant::now);
                if since.elapsed() > STALL_TIMEOUT {
                    Flush::Abandoned
                } else {
                    Flush::Stalled
                }
            }
            // Receiver gone. Dropping a `Generation` sets the cancel flag too,
            // but a closed channel is the earlier and more direct signal.
            Err(mpsc::error::TrySendError::Closed(_)) => Flush::Abandoned,
        }
    }

    fn cancelled(&self) -> bool {
        self.job.cancel.load(Ordering::Acquire)
    }

    fn elapsed_ms(&self) -> u64 {
        ((ggml_time_us() - self.started_us).max(0) / 1_000) as u64
    }
}

enum Flush {
    /// Nothing outstanding; the slot can be decoded.
    Ready,
    /// The caller is behind. Skip this slot for now.
    Stalled,
    /// The caller is gone, or too far behind to wait for.
    Abandoned,
}

struct Runtime {
    model: LlamaModel,
    backend: &'static LlamaBackend,
    info: ModelInfo,
    /// Context available to each slot. The shared context is this times the
    /// slot count.
    n_ctx: u32,
    max_inflight: u32,
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
        Ok(Self {
            model,
            backend,
            info,
            n_ctx: config.n_ctx,
            max_inflight: config.max_inflight.max(1) as u32,
        })
    }

    /// The engine loop. One context, one batch, and one decode per step across
    /// every slot that has work: the weights are read once and every sequence
    /// in the batch rides along on that read, which is the whole point.
    fn serve(
        &self,
        mut jobs: mpsc::Receiver<Job>,
        ready: std::sync::mpsc::Sender<Result<ModelInfo>>,
        inflight: Arc<AtomicUsize>,
    ) {
        // Context is per slot, so the shared window is the sum. A prefill may
        // be as long as one slot's window, so the batch has to hold that many
        // tokens even though a decode step only needs one per slot.
        let total_ctx = self.n_ctx.saturating_mul(self.max_inflight);
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(total_ctx))
            .with_n_seq_max(self.max_inflight)
            .with_n_batch(self.n_ctx.max(512));

        let mut ctx = match self
            .model
            .new_context(self.backend, params)
            .context("failed to create the llama context")
        {
            Ok(ctx) => ctx,
            Err(err) => {
                let _ = ready.send(Err(err));
                return;
            }
        };

        if ready.send(Ok(self.info.clone())).is_err() {
            return; // Caller gave up while we were loading.
        }
        info!(
            slots = self.max_inflight,
            slot_ctx = self.n_ctx,
            total_ctx,
            "engine ready"
        );

        let mut batch = LlamaBatch::new(self.n_ctx.max(512) as usize, self.max_inflight as i32);
        let mut slots: Vec<Option<Slot>> = (0..self.max_inflight).map(|_| None).collect();
        // Terminal events for slots that have already given back their
        // sequence. Kept here so a caller that stopped reading cannot delay
        // the engine, only its own last message.
        let mut outbox: Vec<(mpsc::Sender<Event>, Event)> = Vec::new();
        let mut closed = false;

        loop {
            outbox.retain(|(events, event)| {
                matches!(events.try_send(event.clone()), Err(mpsc::error::TrySendError::Full(_)))
            });

            for index in 0..slots.len() {
                if let Some(slot) = slots[index].as_mut()
                    && matches!(slot.flush(), Flush::Abandoned)
                {
                    warn!(seq_id = slot.seq_id, "caller stopped reading");
                    self.retire(&mut ctx, &mut slots, index, &inflight, &mut outbox, Stop::Cancelled);
                }
            }

            closed |= self.admit(&mut ctx, &mut slots, &mut jobs, &inflight, &mut outbox, &mut batch);

            let active = slots.iter().flatten().count();
            if closed && active == 0 && outbox.is_empty() {
                break;
            }

            match self.step(&mut ctx, &mut slots, &inflight, &mut outbox, &mut batch) {
                Step::Decoded => {}
                // Nothing could be decoded: every slot is waiting on its
                // caller, or there is no work at all and only the outbox is
                // keeping us here.
                Step::Idle => std::thread::sleep(IDLE_TICK),
            }
        }
    }

    /// Fill free slots from the queue. Returns whether the queue has closed.
    ///
    /// Blocks only when there is nothing else to do: with work in flight a
    /// request that has not arrived yet must not delay the ones that have.
    fn admit(
        &self,
        ctx: &mut LlamaContext,
        slots: &mut [Option<Slot>],
        jobs: &mut mpsc::Receiver<Job>,
        inflight: &Arc<AtomicUsize>,
        outbox: &mut Vec<(mpsc::Sender<Event>, Event)>,
        batch: &mut LlamaBatch,
    ) -> bool {
        loop {
            let Some(index) = slots.iter().position(Option::is_none) else {
                return false;
            };
            let idle = slots.iter().all(Option::is_none) && outbox.is_empty();

            let job = if idle {
                match jobs.blocking_recv() {
                    Some(job) => job,
                    None => return true,
                }
            } else {
                match jobs.try_recv() {
                    Ok(job) => job,
                    Err(mpsc::error::TryRecvError::Empty) => return false,
                    Err(mpsc::error::TryRecvError::Disconnected) => return true,
                }
            };

            if job.cancel.load(Ordering::Acquire) {
                // Cancelled while queued: never touch the GPU for it.
                inflight.fetch_sub(1, Ordering::AcqRel);
                outbox.push((
                    job.events.clone(),
                    Event::Done {
                        stop: Stop::Cancelled,
                        prompt_tokens: 0,
                        tokens: 0,
                        elapsed_ms: 0,
                    },
                ));
                continue;
            }

            let seq_id = index as u32;
            match self.prefill(ctx, batch, job, seq_id) {
                Ok(Admitted::Running(slot)) => slots[index] = Some(slot),
                Ok(Admitted::Finished { job, stop, prompt_tokens, elapsed_ms }) => {
                    inflight.fetch_sub(1, Ordering::AcqRel);
                    let _ = ctx.clear_kv_cache_seq(Some(seq_id), None, None);
                    outbox.push((
                        job.events.clone(),
                        Event::Done { stop, prompt_tokens, tokens: 0, elapsed_ms },
                    ));
                }
                Err((job, err)) => {
                    warn!(error = %err, "generation failed");
                    inflight.fetch_sub(1, Ordering::AcqRel);
                    let _ = ctx.clear_kv_cache_seq(Some(seq_id), None, None);
                    outbox.push((job.events.clone(), Event::Failed(err.to_string())));
                }
            }
        }
    }

    /// Ingest a prompt and sample its first token, which is the one thing that
    /// cannot ride along with the other slots: a prefill is many tokens where a
    /// decode step is one. Every running slot waits out this decode.
    #[allow(clippy::type_complexity)]
    fn prefill(
        &self,
        ctx: &mut LlamaContext,
        batch: &mut LlamaBatch,
        job: Job,
        seq_id: u32,
    ) -> std::result::Result<Admitted, (Job, anyhow::Error)> {
        let started = ggml_time_us();
        let prepared = (|| -> Result<(Vec<LlamaToken>, LlamaSampler)> {
            let prompt = self.build_prompt(&job.request)?;
            let tokens = self
                .model
                .str_to_token(&prompt, AddBos::Always)
                .context("failed to tokenize the prompt")?;
            anyhow::ensure!(
                (tokens.len() as u32) < self.n_ctx,
                "prompt is {} tokens and a slot's context is {}",
                tokens.len(),
                self.n_ctx
            );
            let sampler = if job.request.temperature > 0.0 {
                LlamaSampler::chain_simple([
                    LlamaSampler::temp(job.request.temperature),
                    LlamaSampler::top_p(job.request.top_p, 1),
                    LlamaSampler::dist(job.request.seed),
                ])
            } else {
                LlamaSampler::chain_simple([LlamaSampler::greedy()])
            };
            Ok((tokens, sampler))
        })();

        let (tokens, mut sampler) = match prepared {
            Ok(prepared) => prepared,
            Err(err) => return Err((job, err)),
        };

        batch.clear();
        let last = tokens.len() as i32 - 1;
        for (position, token) in (0i32..).zip(tokens.iter().copied()) {
            if let Err(err) = batch.add(token, position, &[seq_id as i32], position == last) {
                return Err((job, err.into()));
            }
        }
        if let Err(err) = ctx.decode(batch).context("prompt decode failed") {
            return Err((job, err));
        }

        let elapsed = (ggml_time_us() - started) as f64 / 1e6;
        info!(
            seq_id,
            prompt_tokens = tokens.len(),
            elapsed_ms = (elapsed * 1e3) as u64,
            tok_per_sec = tokens.len() as f64 / elapsed.max(f64::EPSILON),
            "prompt ingested"
        );

        let token = sampler.sample(ctx, last);
        sampler.accept(token);

        let prompt_tokens = tokens.len() as u32;
        let room = self.n_ctx - prompt_tokens;
        let asked = job.request.max_tokens.max(0) as u32;
        let budget = asked.min(room);

        if self.model.is_eog_token(token) || budget == 0 {
            let stop = if self.model.is_eog_token(token) {
                Stop::EndOfGeneration
            } else {
                Stop::ContextFull
            };
            return Ok(Admitted::Finished {
                job,
                stop,
                prompt_tokens,
                elapsed_ms: (elapsed * 1e3) as u64,
            });
        }

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let piece = match self.model.token_to_piece(token, &mut decoder, false, None) {
            Ok(piece) => piece,
            Err(err) => return Err((job, err.into())),
        };

        Ok(Admitted::Running(Slot {
            job,
            sampler,
            decoder,
            seq_id,
            n_cur: prompt_tokens as i32,
            next_input: token,
            prompt_tokens,
            n_decoded: 1,
            budget,
            context_bound: asked > room,
            started_us: ggml_time_us(),
            pending: Some(Event::Token(piece)),
            stalled_since: None,
        }))
    }

    /// One decode across every slot that is ready, then one token each.
    fn step(
        &self,
        ctx: &mut LlamaContext,
        slots: &mut [Option<Slot>],
        inflight: &Arc<AtomicUsize>,
        outbox: &mut Vec<(mpsc::Sender<Event>, Event)>,
        batch: &mut LlamaBatch,
    ) -> Step {
        batch.clear();
        let mut rows: Vec<(usize, i32)> = Vec::new();

        for index in 0..slots.len() {
            let Some(slot) = slots[index].as_mut() else {
                continue;
            };
            if slot.cancelled() {
                self.retire(ctx, slots, index, inflight, outbox, Stop::Cancelled);
                continue;
            }
            if slot.pending.is_some() {
                continue; // Waiting on its caller, not on the GPU.
            }

            let row = batch.n_tokens();
            if let Err(err) = batch.add(slot.next_input, slot.n_cur, &[slot.seq_id as i32], true) {
                let message = err.to_string();
                self.fail(ctx, slots, index, inflight, outbox, message);
                continue;
            }
            rows.push((index, row));
        }

        if rows.is_empty() {
            return Step::Idle;
        }

        let enqueued = ggml_time_us();
        let decoded = ctx.decode(batch).context("decode failed");
        let enqueue_us = ggml_time_us() - enqueued;
        if let Err(err) = decoded {
            // The batch is shared, so a failed decode belongs to every slot in
            // it. None of them can be trusted to continue.
            let message = err.to_string();
            for (index, _) in rows {
                self.fail(ctx, slots, index, inflight, outbox, message.clone());
            }
            return Step::Decoded;
        }

        let mut gpu_us = 0i64;
        let mut sampled = 0;
        for (index, row) in rows {
            let Some(slot) = slots[index].as_mut() else {
                continue;
            };
            slot.n_cur += 1;

            // Metal runs `decode` asynchronously, so the enqueue above returns
            // long before the work is done and the first read of the logits is
            // what waits for it. Timing that read is how the GPU's real cost
            // shows up; the samples after it are pure CPU.
            let started = ggml_time_us();
            let token = slot.sampler.sample(ctx, row);
            if sampled == 0 {
                gpu_us = ggml_time_us() - started;
            }
            sampled += 1;
            slot.sampler.accept(token);

            if self.model.is_eog_token(token) {
                self.retire(ctx, slots, index, inflight, outbox, Stop::EndOfGeneration);
                continue;
            }

            let piece = match self.model.token_to_piece(token, &mut slot.decoder, false, None) {
                Ok(piece) => piece,
                Err(err) => {
                    let message = err.to_string();
                    self.fail(ctx, slots, index, inflight, outbox, message);
                    continue;
                }
            };
            trace!(seq_id = slot.seq_id, token = token.0, %piece, "token");

            slot.next_input = token;
            slot.n_decoded += 1;
            slot.pending = Some(Event::Token(piece));

            if slot.n_decoded >= slot.budget {
                let stop = if slot.context_bound { Stop::ContextFull } else { Stop::Limit };
                // The slot keeps its pending piece: `retire` hands both it and
                // the terminal event to the outbox, in order.
                self.retire(ctx, slots, index, inflight, outbox, stop);
            }
        }

        debug!(slots = sampled, enqueue_us, gpu_us, "step");
        Step::Decoded
    }

    /// Give back a slot's sequence and tell its caller why.
    ///
    /// The sequence and the count are released before the terminal event goes
    /// out, so a caller that reads `capacity()` on seeing it never sees a slot
    /// that is finished but still counted.
    fn retire(
        &self,
        ctx: &mut LlamaContext,
        slots: &mut [Option<Slot>],
        index: usize,
        inflight: &Arc<AtomicUsize>,
        outbox: &mut Vec<(mpsc::Sender<Event>, Event)>,
        stop: Stop,
    ) {
        let Some(slot) = slots[index].take() else {
            return;
        };
        let elapsed_ms = slot.elapsed_ms();
        let _ = ctx.clear_kv_cache_seq(Some(slot.seq_id), None, None);
        inflight.fetch_sub(1, Ordering::AcqRel);

        info!(
            seq_id = slot.seq_id,
            tokens = slot.n_decoded,
            prompt_tokens = slot.prompt_tokens,
            elapsed_ms,
            tok_per_sec = f64::from(slot.n_decoded) / (elapsed_ms.max(1) as f64 / 1e3),
            ?stop,
            "generation complete"
        );

        if let Some(pending) = slot.pending {
            outbox.push((slot.job.events.clone(), pending));
        }
        outbox.push((
            slot.job.events.clone(),
            Event::Done {
                stop,
                prompt_tokens: slot.prompt_tokens,
                tokens: slot.n_decoded,
                elapsed_ms,
            },
        ));
    }

    fn fail(
        &self,
        ctx: &mut LlamaContext,
        slots: &mut [Option<Slot>],
        index: usize,
        inflight: &Arc<AtomicUsize>,
        outbox: &mut Vec<(mpsc::Sender<Event>, Event)>,
        error: String,
    ) {
        let Some(slot) = slots[index].take() else {
            return;
        };
        warn!(seq_id = slot.seq_id, %error, "generation failed");
        let _ = ctx.clear_kv_cache_seq(Some(slot.seq_id), None, None);
        inflight.fetch_sub(1, Ordering::AcqRel);
        outbox.push((slot.job.events.clone(), Event::Failed(error)));
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

/// What admitting a job produced: a running slot, or an answer already.
enum Admitted {
    Running(Slot),
    Finished { job: Job, stop: Stop, prompt_tokens: u32, elapsed_ms: u64 },
}

enum Step {
    Decoded,
    Idle,
}
