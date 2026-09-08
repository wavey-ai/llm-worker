//! CLI over the llmq engine. Thin by design: it exercises the same interface a
//! worker would use, so the library stays honest.

use std::io::Write;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use llmq::{Engine, EngineConfig, Event, GenerateRequest, Message, ModelSource, Stop};
use tracing_subscriber::EnvFilter;

const HF_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const HF_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";

/// llama.cpp is chatty at info, so it starts a level quieter than our own spans.
const DEFAULT_FILTER: &str = "llmq=info,llama-cpp-2=warn";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Use a local GGUF file instead of the Hugging Face cache.
    #[arg(short, long)]
    model: Option<PathBuf>,

    /// Hugging Face repo to pull the model from.
    #[arg(long, default_value = HF_REPO)]
    repo: String,

    /// File within the repo.
    #[arg(long, default_value = HF_FILE)]
    file: String,

    /// The user message.
    #[arg(short, long, default_value = "Say hi.")]
    prompt: String,

    /// A system message, prepended to the conversation.
    #[arg(long)]
    system: Option<String>,

    /// Maximum tokens to generate.
    #[arg(short = 'n', long, default_value_t = 128)]
    max_tokens: i32,

    /// Context window size.
    #[arg(short, long, default_value_t = 8192)]
    ctx_size: u32,

    /// Sampling temperature. Zero is greedy and deterministic.
    #[arg(short, long, default_value_t = 0.0)]
    temperature: f32,

    /// RNG seed, used when temperature is above zero.
    #[arg(short, long, default_value_t = 42)]
    seed: u32,

    /// Let the model emit its <think> block instead of suppressing it.
    #[arg(long)]
    think: bool,

    /// Cancel automatically after this many tokens, to demonstrate cancellation.
    #[arg(long)]
    cancel_after: Option<u32>,

    /// Log filter, overriding RUST_LOG. Try `debug`, `llmq=trace` for per-token
    /// events, or `llama-cpp-2=debug` for llama.cpp's own output.
    #[arg(short, long)]
    log: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let filter = match &args.log {
        Some(directives) => EnvFilter::new(directives.clone()),
        None => EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(DEFAULT_FILTER)),
    };
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    let model = match &args.model {
        Some(path) => ModelSource::Local(path.clone()),
        None => ModelSource::HuggingFace {
            repo: args.repo.clone(),
            file: args.file.clone(),
        },
    };

    let engine = Engine::spawn(EngineConfig {
        model,
        n_ctx: args.ctx_size,
        ..EngineConfig::default()
    })?;

    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(Message::system(system.clone()));
    }
    messages.push(Message::user(args.prompt.clone()));

    let mut generation = engine.submit(GenerateRequest {
        messages,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        seed: args.seed,
        think: args.think,
        ..GenerateRequest::default()
    })?;

    let mut stdout = std::io::stdout();
    let mut seen: u32 = 0;

    loop {
        tokio::select! {
            // Ctrl-C cancels rather than killing the process, so the engine
            // gets to release the slot.
            _ = tokio::signal::ctrl_c() => {
                eprintln!();
                generation.cancel();
            }
            event = generation.recv() => match event {
                Some(Event::Token(piece)) => {
                    write!(stdout, "{piece}")?;
                    stdout.flush()?;
                    seen += 1;
                    if args.cancel_after == Some(seen) {
                        generation.cancel();
                    }
                }
                Some(Event::Done { stop, prompt_tokens, tokens, elapsed_ms }) => {
                    writeln!(stdout)?;
                    let rate = f64::from(tokens) / (elapsed_ms.max(1) as f64 / 1e3);
                    eprintln!(
                        "{prompt_tokens} prompt, {tokens} tokens in {elapsed_ms}ms ({rate:.1} tok/s), stop: {stop:?}"
                    );
                    if stop == Stop::Cancelled {
                        eprintln!("cancelled; slot released");
                    }
                    break;
                }
                Some(Event::Failed(error)) => {
                    writeln!(stdout)?;
                    anyhow::bail!("{error}");
                }
                None => break,
            },
        }
    }

    Ok(())
}
