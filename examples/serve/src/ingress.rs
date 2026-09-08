//! The `--serve` mode: this process is also the ingress.
//!
//! The ring already knows how to take an HTTP request, park it, and stream a
//! worker's answer back out — `UploadResponseRouter` does the whole job. All
//! this router adds is a health endpoint, a model list, and a decision about
//! which paths are allowed through to the ring at all.

use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use http::{Method, Request, StatusCode};
use llm_engine::Engine;
use upload_response::UploadResponseRouter;
use web_service::{
    BodyStream, HandlerResponse, HandlerResult, Router, ServerError, StreamWriter,
    WebSocketHandler, WebTransportHandler,
};

pub const CHAT_COMPLETIONS_PATH: &str = "/v1/chat/completions";

pub struct AppRouter {
    upload: Arc<UploadResponseRouter>,
    engine: Engine,
    model_name: String,
}

impl AppRouter {
    pub fn new(upload: Arc<UploadResponseRouter>, engine: Engine, model_name: String) -> Self {
        Self { upload, engine, model_name }
    }

    fn is_chat_path(path: &str) -> bool {
        path == CHAT_COMPLETIONS_PATH
    }

    fn json(status: StatusCode, body: String) -> HandlerResponse {
        HandlerResponse {
            status,
            body: Some(Bytes::from(body)),
            content_type: Some("application/json".into()),
            headers: vec![("cache-control".into(), "no-store".into())],
            etag: None,
        }
    }

    fn not_found() -> HandlerResponse {
        Self::json(
            StatusCode::NOT_FOUND,
            llm_worker::protocol::error_body("not found", "invalid_request_error"),
        )
    }

    /// Health reports the engine's own occupancy rather than the ring's, so
    /// an operator sees the number the GPU is actually working from.
    fn health(&self) -> HandlerResponse {
        let capacity = self.engine.capacity();
        let info = self.engine.model_info();
        Self::json(
            StatusCode::OK,
            serde_json::json!({
                "status": "ok",
                "model": self.model_name,
                "model_path": info.path.display().to_string(),
                "n_params": info.n_params,
                "n_ctx": info.n_ctx,
                "inflight": capacity.inflight,
                "max_inflight": capacity.max_inflight,
                "available_slots": capacity.available_slots,
            })
            .to_string(),
        )
    }

    fn models(&self) -> HandlerResponse {
        Self::json(
            StatusCode::OK,
            serde_json::json!({
                "object": "list",
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "llm",
                }],
            })
            .to_string(),
        )
    }
}

#[async_trait]
impl Router for AppRouter {
    async fn route(&self, req: Request<()>) -> HandlerResult<HandlerResponse> {
        match (req.method(), req.uri().path()) {
            (&Method::GET, "/") | (&Method::GET, "/health") | (&Method::GET, "/healthz") => {
                Ok(self.health())
            }
            (&Method::GET, "/v1/models") => Ok(self.models()),
            _ => Ok(Self::not_found()),
        }
    }

    async fn route_body(
        &self,
        req: Request<()>,
        body: BodyStream,
    ) -> HandlerResult<HandlerResponse> {
        let _ = body;
        if Self::is_chat_path(req.uri().path()) {
            // Reached only if a backend declines the combined path below.
            return Err(ServerError::Config(
                "chat completions need streaming request and response handling".into(),
            ));
        }
        Ok(Self::not_found())
    }

    fn has_body_handler(&self, path: &str) -> bool {
        Self::is_chat_path(path)
    }

    fn has_body_stream_handler(&self, path: &str) -> bool {
        Self::is_chat_path(path)
    }

    async fn route_body_stream(
        &self,
        req: Request<()>,
        body: BodyStream,
        stream_writer: Box<dyn StreamWriter>,
    ) -> HandlerResult<()> {
        if !Self::is_chat_path(req.uri().path()) {
            return Err(ServerError::Config("no streaming handler for this route".into()));
        }
        // The ring owns the whole exchange from here: it parks the request,
        // a worker claims it, and the response lane is proxied straight back
        // to this client. When the client goes away the stream is closed,
        // which is what makes the worker's next write fail and its generation
        // cancel.
        self.upload.route_body_stream(req, body, stream_writer).await
    }

    fn is_streaming(&self, _path: &str) -> bool {
        false
    }

    async fn route_stream(
        &self,
        _req: Request<()>,
        _stream_writer: Box<dyn StreamWriter>,
    ) -> HandlerResult<()> {
        Err(ServerError::Config("stream-only routes are not supported".into()))
    }

    fn webtransport_handler(&self) -> Option<&dyn WebTransportHandler> {
        None
    }

    fn websocket_handler(&self, _path: &str) -> Option<&dyn WebSocketHandler> {
        None
    }
}
