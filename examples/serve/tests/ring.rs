//! End to end over the ring, in one process: a request goes into the
//! router, a worker claims it, and the engine's tokens come back out of the
//! response lane as they are produced.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use futures_util::StreamExt;
use http::{Request, Response, StatusCode};
use llm_engine::{Engine, EngineConfig};
use llm_serve::ingress::{AppRouter, CHAT_COMPLETIONS_PATH};
use llm_worker::worker::{LlmWorker, WorkerConfig};
use upload_response::{UploadResponseConfig, UploadResponseRouter, UploadResponseService};
use web_service::{BodyStream, Router, ServerError, StreamWriter};

/// What the peer would have seen.
#[derive(Debug, Default)]
struct Collected {
    status: Option<StatusCode>,
    content_type: Option<String>,
    body: Vec<u8>,
    finished: bool,
}

impl Collected {
    fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).to_string()
    }
}

struct Collector(Arc<Mutex<Collected>>);

#[async_trait]
impl StreamWriter for Collector {
    async fn send_response(&mut self, response: Response<()>) -> Result<(), ServerError> {
        let mut collected = self.0.lock().unwrap();
        collected.status = Some(response.status());
        collected.content_type = response
            .headers()
            .get(http::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_string);
        Ok(())
    }

    async fn send_data(&mut self, data: Bytes) -> Result<(), ServerError> {
        self.0.lock().unwrap().body.extend_from_slice(&data);
        Ok(())
    }

    async fn finish(&mut self) -> Result<(), ServerError> {
        self.0.lock().unwrap().finished = true;
        Ok(())
    }
}

/// One engine, one worker, one router, wired the way `--serve` wires them.
/// The engine comes back too, for tests that watch occupancy.
fn harness() -> (AppRouter, Engine) {
    let engine = Engine::spawn(EngineConfig { native_logs: false, ..EngineConfig::default() })
        .expect("engine failed to start");
    let service = Arc::new(UploadResponseService::new(UploadResponseConfig::default()));

    let mut config = WorkerConfig::new("llm-test");
    config.poll_interval = Duration::from_millis(10);
    let worker = Arc::new(LlmWorker::new(engine.clone(), config));
    worker.spawn_local(Arc::clone(&service));

    let router = AppRouter::new(
        Arc::new(UploadResponseRouter::new(service)),
        engine.clone(),
        "llm-test".to_string(),
    );
    (router, engine)
}

async fn post(router: &AppRouter, path: &str, body: &str) -> Collected {
    let request = Request::builder()
        .method("POST")
        .uri(path)
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(())
        .unwrap();
    let body: BodyStream = futures_util::stream::once({
        let body = Bytes::copy_from_slice(body.as_bytes());
        async move { Ok(body) }
    })
    .boxed();

    let collected = Arc::new(Mutex::new(Collected::default()));
    let writer = Box::new(Collector(Arc::clone(&collected)));

    tokio::time::timeout(
        Duration::from_secs(90),
        router.route_body_stream(request, body, writer),
    )
    .await
    .expect("request did not complete")
    .expect("routing failed");

    Arc::try_unwrap(collected).unwrap().into_inner().unwrap()
}

/// Frames of an SSE body, `data: ` stripped.
fn frames(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| {
            frame
                .strip_prefix("data: ")
                .expect("frame is an SSE data line")
                .to_string()
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread")]
async fn streams_a_completion_back_through_the_ring() {
    let (router, _engine) = harness();
    let collected = post(
        &router,
        CHAT_COMPLETIONS_PATH,
        r#"{"model":"llm-test","messages":[{"role":"user","content":"Say hi."}],
            "stream":true,"max_tokens":32,"stream_options":{"include_usage":true}}"#,
    )
    .await;

    assert_eq!(collected.status, Some(StatusCode::OK));
    assert_eq!(
        collected.content_type.as_deref(),
        Some("text/event-stream; charset=utf-8")
    );
    assert!(collected.finished, "the response was never finished");

    let frames = frames(&collected.text());
    assert_eq!(frames.last().unwrap(), "[DONE]");

    let open: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
    assert_eq!(open["choices"][0]["delta"]["role"], "assistant");

    let text: String = frames[1..frames.len() - 2]
        .iter()
        .map(|frame| {
            let chunk: serde_json::Value = serde_json::from_str(frame).unwrap();
            chunk["choices"][0]["delta"]["content"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    assert!(!text.is_empty(), "no text was streamed");
    assert!(!text.contains("<think>"), "reasoning block leaked: {text}");

    let close: serde_json::Value =
        serde_json::from_str(&frames[frames.len() - 2]).unwrap();
    assert!(close["choices"][0]["finish_reason"].is_string());
    assert!(
        close["usage"]["completion_tokens"].as_u64().unwrap() > 0,
        "usage was requested but not reported"
    );
    assert!(close["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn answers_a_non_streaming_request_with_one_json_body() {
    let (router, _engine) = harness();
    let collected = post(
        &router,
        CHAT_COMPLETIONS_PATH,
        r#"{"messages":[{"role":"user","content":"Say hi."}],"max_tokens":32}"#,
    )
    .await;

    assert_eq!(collected.status, Some(StatusCode::OK));
    assert_eq!(collected.content_type.as_deref(), Some("application/json"));

    let completion: serde_json::Value = serde_json::from_str(&collected.text()).unwrap();
    assert_eq!(completion["object"], "chat.completion");
    let content = completion["choices"][0]["message"]["content"]
        .as_str()
        .expect("a message");
    assert!(!content.is_empty());
    assert!(completion["usage"]["completion_tokens"].as_u64().unwrap() > 0);
}

/// A request the worker cannot understand must come back as an error to the
/// caller, not as a stream that never starts.
#[tokio::test(flavor = "multi_thread")]
async fn rejects_a_malformed_request() {
    let (router, _engine) = harness();
    let collected = post(&router, CHAT_COMPLETIONS_PATH, r#"{"messages":[]}"#).await;

    assert_eq!(collected.status, Some(StatusCode::BAD_REQUEST));
    let error: serde_json::Value = serde_json::from_str(&collected.text()).unwrap();
    assert_eq!(error["error"]["type"], "invalid_request_error");
}

/// Several clients at once. The engine is single-slot, so this is a queue and
/// not a batch: the ring parks the waiting requests, the engine never holds
/// more than its slot count, and every caller still gets a whole answer of
/// its own. When the multi-slot loop lands, only the bound changes.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_requests_queue_without_crossing_streams() {
    const CLIENTS: usize = 4;

    let (router, engine) = harness();
    let slots = engine.capacity().max_inflight;
    let router = Arc::new(router);

    // Sampling can miss a peak, so this can only ever understate occupancy —
    // it will not fail a build for being unlucky.
    let peak = Arc::new(AtomicUsize::new(0));
    let watching = Arc::new(AtomicBool::new(true));
    let sampler = tokio::spawn({
        let (engine, peak, watching) = (engine.clone(), Arc::clone(&peak), Arc::clone(&watching));
        async move {
            while watching.load(Ordering::Relaxed) {
                peak.fetch_max(engine.capacity().inflight, Ordering::Relaxed);
                tokio::task::yield_now().await;
            }
        }
    });

    let mut clients = Vec::new();
    for client in 0..CLIENTS {
        let router = Arc::clone(&router);
        clients.push(tokio::spawn(async move {
            post(
                &router,
                CHAT_COMPLETIONS_PATH,
                &format!(
                    r#"{{"messages":[{{"role":"user","content":"Say hi, request {client}."}}],
                        "stream":true,"max_tokens":16}}"#
                ),
            )
            .await
        }));
    }

    let mut ids = std::collections::HashSet::new();
    for client in clients {
        let collected = client.await.expect("a client panicked");
        assert_eq!(collected.status, Some(StatusCode::OK));
        assert!(collected.finished);

        let frames = frames(&collected.text());
        assert_eq!(frames.last().unwrap(), "[DONE]");

        let open: serde_json::Value = serde_json::from_str(&frames[0]).unwrap();
        let id = open["id"].as_str().expect("a completion id").to_string();
        assert!(ids.insert(id), "two clients were served the same generation");

        // Every frame in this response must belong to this response.
        for frame in &frames[..frames.len() - 1] {
            let chunk: serde_json::Value = serde_json::from_str(frame).unwrap();
            assert_eq!(chunk["id"], open["id"], "a chunk leaked between streams");
        }
    }

    watching.store(false, Ordering::Relaxed);
    let _ = sampler.await;

    assert_eq!(ids.len(), CLIENTS);
    assert!(
        peak.load(Ordering::Relaxed) <= slots,
        "engine held {} requests with {slots} slot(s): the queue belongs in the ring",
        peak.load(Ordering::Relaxed)
    );
    assert_eq!(engine.capacity().inflight, 0, "a slot was not released");
}

#[tokio::test(flavor = "multi_thread")]
async fn serves_the_page_and_health_without_touching_the_ring() {
    let (router, _engine) = harness();

    let request = Request::builder().method("GET").uri("/health").body(()).unwrap();
    let response = router.route(request).await.expect("health should answer");
    assert_eq!(response.status, StatusCode::OK);
    let health: serde_json::Value =
        serde_json::from_slice(&response.body.expect("a body")).unwrap();
    assert_eq!(health["status"], "ok");
    assert_eq!(health["model"], "llm-test");
    assert_eq!(health["inflight"], 0);

    let request = Request::builder().method("GET").uri("/").body(()).unwrap();
    let response = router.route(request).await.expect("the page should answer");
    assert_eq!(response.status, StatusCode::OK);
    assert_eq!(response.content_type.as_deref(), Some("text/html; charset=utf-8"));
    let page = String::from_utf8(response.body.expect("a body").to_vec()).unwrap();
    assert!(page.contains(CHAT_COMPLETIONS_PATH), "the page must post somewhere real");
}
