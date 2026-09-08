//! llm-worker: an engine attached to the upload-response ring.
//!
//! The GPU sits here; the front door is somewhere else. This process holds no
//! listening socket — it discovers ingress services, claims request lanes over
//! HTTP, and writes tokens back down response lanes.
//!
//! For a front door of your own, `examples/serve` runs one in-process.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use clap::Parser;
use llm_engine::{Engine, EngineConfig, ModelSource};
use tokio::time::Duration;
use tracing::info;
use tracing_subscriber::EnvFilter;
use upload_response::{RemoteIngressClient, UploadResponseConfig};

use llm_worker::protocol::{DEFAULT_MODEL, RequestDefaults};
use llm_worker::worker::{LlmWorker, WorkerConfig};

const HF_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const HF_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";

const DEFAULT_FILTER: &str =
    "llm_engine=info,llm_worker=info,gpu_worker_upload_response=info,llama-cpp-2=warn";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Use a local GGUF file instead of the Hugging Face cache.
    #[arg(short, long, env = "LLMQ_MODEL")]
    model: Option<PathBuf>,

    /// Hugging Face repo to pull the model from.
    #[arg(long, default_value = HF_REPO)]
    repo: String,

    /// File within the repo.
    #[arg(long, default_value = HF_FILE)]
    file: String,

    /// Name this worker reports as the model in completions.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model_name: String,

    /// Context window size.
    #[arg(short, long, default_value_t = 8192)]
    ctx_size: u32,

    /// Identifies this worker to the ring. Must be unique across workers.
    #[arg(long, env = "LLMQ_WORKER_ID")]
    worker_id: Option<String>,

    /// Completions generated at once. The engine is single-slot for now, so
    /// anything above one only moves the queue from the ring into the engine.
    #[arg(long, default_value_t = 1)]
    max_inflight: usize,

    /// Ingress services to pull work from. Repeatable.
    #[arg(long = "ingress-url", env = "LLMQ_INGRESS_URLS", value_delimiter = ',')]
    ingress_urls: Vec<String>,

    /// DNS name whose A records are ingress services.
    #[arg(long, env = "LLMQ_DISCOVERY_DNS")]
    discovery_dns: Option<String>,

    /// Accept ingress certificates that do not validate. Development only.
    #[arg(long)]
    insecure_tls: bool,

    /// Ring slot size. Must match the ingress this worker talks to, since it
    /// is how a body is cut into slots.
    #[arg(long, default_value_t = 32)]
    slot_size_kb: usize,

    /// How often to poll a lane for the next slot.
    #[arg(long, default_value_t = 50)]
    poll_ms: u64,

    /// How often to publish capacity to the ring.
    #[arg(long, default_value_t = 1000)]
    heartbeat_ms: u64,

    /// How often to refresh the ingress origin list.
    #[arg(long, default_value_t = 5000)]
    discovery_ms: u64,

    /// How long a request body may take to arrive before the job is dropped.
    #[arg(long, default_value_t = 30_000)]
    request_timeout_ms: u64,

    /// Default max tokens, when a request does not ask for one.
    #[arg(long, default_value_t = 512)]
    default_max_tokens: i32,

    /// Default sampling temperature. Zero is greedy and deterministic.
    #[arg(long, default_value_t = 0.0)]
    default_temperature: f32,

    /// Default nucleus sampling cutoff.
    #[arg(long, default_value_t = 0.9)]
    default_top_p: f32,

    /// Default RNG seed, used when temperature is above zero.
    #[arg(long, default_value_t = 42)]
    default_seed: u32,

    /// Let models emit their reasoning block unless a request says otherwise.
    #[arg(long)]
    default_think: bool,

    /// Log filter, overriding RUST_LOG.
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

    if args.ingress_urls.is_empty() && args.discovery_dns.is_none() {
        bail!("nothing to attach to: pass --ingress-url or --discovery-dns");
    }

    let model = match &args.model {
        Some(path) => ModelSource::Local(path.clone()),
        None => ModelSource::HuggingFace { repo: args.repo.clone(), file: args.file.clone() },
    };

    // Blocking until the model is resident is the point: the worker must not
    // publish capacity it cannot honour.
    let engine = Engine::spawn(EngineConfig {
        model,
        n_ctx: args.ctx_size,
        max_inflight: args.max_inflight.max(1),
        ..EngineConfig::default()
    })?;
    info!(
        model = %engine.model_info().path.display(),
        n_params = engine.model_info().n_params,
        n_ctx = engine.model_info().n_ctx,
        "engine ready"
    );

    let worker_id = args
        .worker_id
        .clone()
        .unwrap_or_else(|| format!("llm-{}", std::process::id()));

    let worker = Arc::new(LlmWorker::new(
        engine.clone(),
        WorkerConfig {
            worker_id: worker_id.clone(),
            // The engine is the only authority on how many slots exist; the
            // ring hears its number rather than a second copy of the flag.
            max_inflight: engine.capacity().max_inflight,
            poll_interval: Duration::from_millis(args.poll_ms.max(1)),
            heartbeat_interval: Duration::from_millis(args.heartbeat_ms.max(1)),
            discovery_interval: Duration::from_millis(args.discovery_ms.max(1)),
            request_timeout: Duration::from_millis(args.request_timeout_ms.max(1)),
            ingress_urls: args.ingress_urls.clone(),
            discovery_dns: args.discovery_dns.clone(),
            model_name: args.model_name.clone(),
            defaults: RequestDefaults {
                max_tokens: args.default_max_tokens,
                temperature: args.default_temperature,
                top_p: args.default_top_p,
                seed: args.default_seed,
                think: args.default_think,
            },
        },
    ));

    let slot_bytes = UploadResponseConfig {
        slot_size_kb: args.slot_size_kb.max(1),
        ..UploadResponseConfig::default()
    }
    .slot_bytes();

    let client = RemoteIngressClient::new(slot_bytes, args.insecure_tls)
        .context("failed to build the ingress client")?;
    info!(
        worker_id = %worker_id,
        ingress_urls = ?args.ingress_urls,
        discovery_dns = ?args.discovery_dns,
        max_inflight = engine.capacity().max_inflight,
        "attaching to remote ingress"
    );
    let worker = worker.spawn_remote(client);

    tokio::signal::ctrl_c().await?;
    info!("shutdown signal received");
    worker.abort();
    Ok(())
}
