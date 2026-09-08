//! llmq-serve: the front door and the worker in one process.
//!
//! `llmq-worker` has no listening socket — it attaches to an ingress that
//! already exists. This example is that ingress, running an in-process ring
//! so one binary is enough to try the thing out:
//!
//! ```text
//! llmq-serve --model models/model.gguf --tls-cert cert.pem --tls-key key.pem
//! ```
//!
//! Everything the server crate touches lives here rather than in the worker
//! library, which is why the library can serve a remote ring and this one
//! with the same code.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as BASE64;
use clap::Parser;
use llmq::{Engine, EngineConfig, ModelSource};
use tokio::time::Duration;
use tracing::info;
use tracing_subscriber::EnvFilter;
use upload_response::{UploadResponseConfig, UploadResponseRouter, UploadResponseService};
use web_service::{H2H3Server, Server, ServerBuilder};

use llmq_worker::protocol::DEFAULT_MODEL;
use llmq_worker::worker::{LlmWorker, WorkerConfig};

use llmq_serve::ingress::AppRouter;

const HF_REPO: &str = "unsloth/Qwen3.5-0.8B-GGUF";
const HF_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";

const DEFAULT_FILTER: &str =
    "llmq=info,llmq_worker=info,llmq_serve=info,gpu_worker_upload_response=info,llama-cpp-2=warn";

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

    /// Name reported as the model in completions.
    #[arg(long, default_value = DEFAULT_MODEL)]
    model_name: String,

    /// Context window size.
    #[arg(short, long, default_value_t = 8192)]
    ctx_size: u32,

    /// Completions generated at once.
    #[arg(long, default_value_t = 1)]
    max_inflight: usize,

    #[arg(short, long, default_value_t = 8443)]
    port: u16,

    /// PEM certificate chain.
    #[arg(long, env = "LLMQ_TLS_CERT")]
    tls_cert: Option<PathBuf>,

    /// PEM private key.
    #[arg(long, env = "LLMQ_TLS_KEY")]
    tls_key: Option<PathBuf>,

    #[arg(long)]
    enable_h3: bool,

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

    let model = match &args.model {
        Some(path) => ModelSource::Local(path.clone()),
        None => ModelSource::HuggingFace { repo: args.repo.clone(), file: args.file.clone() },
    };

    // Load before listening: the port opens only once the model can be served.
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

    let service = Arc::new(UploadResponseService::new(UploadResponseConfig::default()));

    let mut config = WorkerConfig::new(format!("llmq-{}", std::process::id()));
    config.max_inflight = engine.capacity().max_inflight;
    config.poll_interval = Duration::from_millis(10);
    config.model_name = args.model_name.clone();
    let worker = Arc::new(LlmWorker::new(engine.clone(), config));

    // The worker reads the same service the router writes into, so a request
    // never leaves this process.
    let worker = worker.spawn_local(Arc::clone(&service));

    let router = Box::new(AppRouter::new(
        Arc::new(UploadResponseRouter::new(service)),
        engine,
        args.model_name.clone(),
    ));

    let (cert, key) = tls_base64(&args)?;
    let server = H2H3Server::builder()
        .with_tls(cert, key)
        .with_port(args.port)
        .enable_h2(true)
        .enable_h3(args.enable_h3)
        .enable_websocket(false)
        .with_router(router)
        .build()
        .context("failed to build the ingress server")?;

    let mut handle = server.start().await.context("failed to start the ingress server")?;
    let ready = std::mem::replace(&mut handle.ready_rx, tokio::sync::oneshot::channel().1);
    let _ = ready.await;
    info!(port = args.port, enable_h3 = args.enable_h3, "ingress ready");

    tokio::signal::ctrl_c().await?;
    info!("shutdown signal received");
    let _ = handle.shutdown_tx.send(());
    let _ = handle.finished_rx.await;
    worker.abort();
    Ok(())
}

/// The server takes PEM as base64, which is how it is carried in deployment
/// config. On the command line a path is friendlier, so read and re-encode.
fn tls_base64(args: &Args) -> Result<(String, String)> {
    let (Some(cert), Some(key)) = (&args.tls_cert, &args.tls_key) else {
        bail!("serving needs --tls-cert and --tls-key");
    };
    let cert =
        std::fs::read(cert).with_context(|| format!("failed to read {}", cert.display()))?;
    let key = std::fs::read(key).with_context(|| format!("failed to read {}", key.display()))?;
    Ok((BASE64.encode(cert), BASE64.encode(key)))
}
