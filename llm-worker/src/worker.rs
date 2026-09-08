//! The worker: one engine, serving whatever the ring hands it.
//!
//! The ring gives out jobs two ways — from an in-process service, or over
//! HTTP from a remote ingress — and the two job types differ only in how a
//! request is read and where a response is written. [`JobLane`] names that
//! difference so the generation path is written once.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use gpu_worker::upload_response::{
    LocalJob, LocalJobProcessor, LocalWorkerConfig, PipelineSpec, RemoteJob, RemoteJobProcessor,
    RemoteWorkerConfig, SinkLane, SourceFrame, SourceLane, run_local_worker_loop,
    run_remote_worker_loop,
};
use http::{Request, StatusCode};
use llm_engine::{Engine, Event};
use tokio::time::timeout;
use tracing::{Instrument, debug, info, info_span, warn};
use upload_response::{RemoteIngressClient, ResponseCacheWriter, UploadResponseService};

use crate::protocol::{
    ChatRequest, JSON_CONTENT_TYPE, RequestDefaults, ResponseWriter, SSE_CONTENT_TYPE, Usage,
    error_body, error_frame,
};

/// The lane a worker claims. Requests arrive on the request lane and the
/// answer goes straight to the response lane: no intermediate stage, because
/// there is nothing between reading a prompt and writing its tokens.
pub fn pipeline_spec() -> PipelineSpec {
    PipelineSpec { source: SourceLane::Request, sink: SinkLane::Response }
}

#[derive(Debug, Clone)]
pub struct WorkerConfig {
    pub worker_id: String,
    /// Requests served at once. This is the engine's own
    /// `Capacity::max_inflight`, passed through to the ring's heartbeat
    /// unchanged; claiming more work than the engine has slots for would
    /// only move the queue out of the ring, where a scheduler can see it,
    /// and into the engine, where it cannot.
    pub max_inflight: usize,
    pub poll_interval: Duration,
    pub heartbeat_interval: Duration,
    pub discovery_interval: Duration,
    /// How long to wait for a request body to finish arriving.
    pub request_timeout: Duration,
    pub ingress_urls: Vec<String>,
    pub discovery_dns: Option<String>,
    pub model_name: String,
    pub defaults: RequestDefaults,
}

impl WorkerConfig {
    pub fn new(worker_id: impl Into<String>) -> Self {
        Self {
            worker_id: worker_id.into(),
            max_inflight: 1,
            poll_interval: Duration::from_millis(50),
            heartbeat_interval: Duration::from_secs(1),
            discovery_interval: Duration::from_secs(5),
            request_timeout: Duration::from_secs(30),
            ingress_urls: Vec::new(),
            discovery_dns: None,
            model_name: crate::protocol::DEFAULT_MODEL.to_string(),
            defaults: RequestDefaults::default(),
        }
    }
}

pub struct LlmWorker {
    engine: Engine,
    config: WorkerConfig,
}

impl LlmWorker {
    pub fn new(engine: Engine, config: WorkerConfig) -> Self {
        Self { engine, config }
    }

    pub fn spawn_local(self: Arc<Self>, service: Arc<UploadResponseService>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let config = self.local_worker_config();
            run_local_worker_loop(service, config, self).await;
        })
    }

    pub fn spawn_remote(self: Arc<Self>, client: RemoteIngressClient) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            let config = self.remote_worker_config();
            run_remote_worker_loop(client, config, self).await;
        })
    }

    fn local_worker_config(&self) -> LocalWorkerConfig {
        let mut config = LocalWorkerConfig::new(self.config.worker_id.clone(), pipeline_spec());
        config.max_inflight = self.config.max_inflight;
        config.poll_interval = self.config.poll_interval;
        config.heartbeat_interval = self.config.heartbeat_interval;
        config
    }

    fn remote_worker_config(&self) -> RemoteWorkerConfig {
        let mut config = RemoteWorkerConfig::new(self.config.worker_id.clone(), pipeline_spec());
        config.max_inflight = self.config.max_inflight;
        config.poll_interval = self.config.poll_interval;
        config.heartbeat_interval = self.config.heartbeat_interval;
        config.discovery_interval = self.config.discovery_interval;
        config.ingress_urls = self.config.ingress_urls.clone();
        config.discovery_dns = self.config.discovery_dns.clone();
        config
    }

    /// Read one request, generate, write the answer back. Every error after
    /// the request is understood is reported to the client rather than only
    /// logged: a caller waiting on a response lane deserves an answer.
    async fn serve(&self, lane: &dyn JobLane) -> Result<()> {
        let mut writer = lane.writer();

        let request = match self.read_request(lane).await {
            Ok(request) => request,
            Err(error) => {
                warn!(error = %error, "rejected request");
                return write_error(&mut writer, StatusCode::BAD_REQUEST, &error.to_string()).await;
            }
        };

        let response = ResponseWriter::new(next_completion_id(), request.model_name().to_string());
        let generate = request.to_generate_request(self.config.defaults);
        let mut generation = match self.engine.submit(generate) {
            Ok(generation) => generation,
            Err(error) => {
                warn!(error = %error, "engine refused the request");
                return write_error(&mut writer, StatusCode::SERVICE_UNAVAILABLE, &error.to_string())
                    .await;
            }
        };

        if request.stream {
            let head = http::Response::builder()
                .status(StatusCode::OK)
                .header(http::header::CONTENT_TYPE, SSE_CONTENT_TYPE)
                .header("cache-control", "no-store")
                .body(())?;
            writer.ensure_started(head).await?;
            writer.send_body(Bytes::from(response.stream_open())).await?;
        }

        let mut text = String::new();
        loop {
            let Some(event) = generation.recv().await else {
                return Err(anyhow!("engine closed the stream without finishing"));
            };
            match event {
                Event::Token(piece) => {
                    if request.stream {
                        // A failed write means the ring closed this stream,
                        // which means the client is gone. Returning here drops
                        // the generation, and dropping it cancels the work.
                        writer.send_body(Bytes::from(response.stream_token(&piece))).await?;
                    } else {
                        text.push_str(&piece);
                    }
                }
                Event::Done { stop, prompt_tokens, tokens, elapsed_ms } => {
                    let usage = Usage::new(prompt_tokens, tokens);
                    info!(
                        completion_id = response.id(),
                        prompt_tokens,
                        tokens,
                        elapsed_ms,
                        ?stop,
                        streaming = request.stream,
                        "served completion"
                    );
                    if request.stream {
                        let usage = request.include_usage().then_some(usage);
                        writer.send_body(Bytes::from(response.stream_close(stop, usage))).await?;
                        return writer.finish().await;
                    }
                    let body = response.completion(text, stop, usage);
                    return write_json(&mut writer, StatusCode::OK, body).await;
                }
                Event::Failed(error) => {
                    warn!(completion_id = response.id(), %error, "generation failed");
                    if request.stream {
                        writer.send_body(Bytes::from(error_frame(&error))).await?;
                        writer.finish().await?;
                    } else {
                        write_error(&mut writer, StatusCode::INTERNAL_SERVER_ERROR, &error).await?;
                    }
                    return Err(anyhow!("{error}"));
                }
            }
        }
    }

    async fn read_request(&self, lane: &dyn JobLane) -> Result<ChatRequest> {
        let head = lane
            .request_head()
            .await?
            .ok_or_else(|| anyhow!("request headers were missing"))?;
        debug!(method = %head.method(), path = %head.uri().path(), "claimed request");

        let body = timeout(self.config.request_timeout, lane.read_body(self.config.poll_interval))
            .await
            .map_err(|_| {
                anyhow!(
                    "request body did not finish within {}ms",
                    self.config.request_timeout.as_millis()
                )
            })??;

        ChatRequest::parse(&body)
    }
}

#[async_trait]
impl LocalJobProcessor for LlmWorker {
    async fn process(&self, job: LocalJob) -> Result<()> {
        let span = info_span!(
            "completion",
            stream_id = job.stream_id,
            worker_id = %job.worker_id(),
            source = "local",
        );
        self.serve(&job).instrument(span).await
    }
}

#[async_trait]
impl RemoteJobProcessor for LlmWorker {
    async fn process(&self, job: RemoteJob) -> Result<()> {
        let span = info_span!(
            "completion",
            stream_id = job.stream_id,
            worker_id = %job.worker_id(),
            origin = %job.origin,
            source = "remote",
        );
        self.serve(&job).instrument(span).await
    }
}

/// The two ways in and out of the ring, reduced to what a completion needs.
#[async_trait]
trait JobLane: Send + Sync {
    async fn request_head(&self) -> Result<Option<Request<()>>>;
    /// The whole request body. Prompts are small and the engine needs all of
    /// it before it can start, so there is nothing to gain from streaming.
    async fn read_body(&self, poll: Duration) -> Result<Bytes>;
    fn writer(&self) -> ResponseCacheWriter;
}

#[async_trait]
impl JobLane for LocalJob {
    async fn request_head(&self) -> Result<Option<Request<()>>> {
        self.request().await
    }

    async fn read_body(&self, poll: Duration) -> Result<Bytes> {
        let mut reader = self.source_reader_from(1, poll);
        let mut body = BytesMut::new();
        while let Some(frame) = reader.next_frame().await? {
            match frame {
                SourceFrame::Body(chunk) => body.extend_from_slice(&chunk),
                SourceFrame::End => break,
                _ => {}
            }
        }
        Ok(body.freeze())
    }

    fn writer(&self) -> ResponseCacheWriter {
        ResponseCacheWriter::local(Arc::clone(self.service()), self.stream_id, self.slot_bytes())
    }
}

#[async_trait]
impl JobLane for RemoteJob {
    async fn request_head(&self) -> Result<Option<Request<()>>> {
        self.request().await
    }

    async fn read_body(&self, poll: Duration) -> Result<Bytes> {
        let mut reader = self.source_reader_from(1, poll);
        let mut body = BytesMut::new();
        while let Some(frame) = reader.next_frame().await? {
            match frame {
                SourceFrame::Body(chunk) => body.extend_from_slice(&chunk),
                SourceFrame::End => break,
                _ => {}
            }
        }
        Ok(body.freeze())
    }

    fn writer(&self) -> ResponseCacheWriter {
        ResponseCacheWriter::remote(
            self.client().clone(),
            self.origin.clone(),
            self.stream_id,
            self.slot_bytes(),
        )
    }
}

/// One JSON body, written as head, body, end. The three-step form is the
/// same one the streaming path uses, which is why this module needs nothing
/// from the server crate to answer a request.
async fn write_json(
    writer: &mut ResponseCacheWriter,
    status: StatusCode,
    body: String,
) -> Result<()> {
    let head = http::Response::builder()
        .status(status)
        .header(http::header::CONTENT_TYPE, JSON_CONTENT_TYPE)
        .header("cache-control", "no-store")
        .body(())?;
    writer.ensure_started(head).await?;
    writer.send_body(Bytes::from(body)).await?;
    writer.finish().await
}

async fn write_error(
    writer: &mut ResponseCacheWriter,
    status: StatusCode,
    message: &str,
) -> Result<()> {
    let kind = match status {
        StatusCode::BAD_REQUEST => "invalid_request_error",
        StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
        _ => "server_error",
    };
    write_json(writer, status, error_body(message, kind)).await
}

/// Unique within a process and ordered, which is all an id has to be.
fn next_completion_id() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as u64)
        .unwrap_or(0);
    let sequence = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("chatcmpl-{started:x}{sequence:04x}")
}
