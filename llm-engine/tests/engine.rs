//! Engine behaviour a worker depends on: streaming, cancellation, and that a
//! cancelled request leaves the engine able to serve the next one.

use std::time::Duration;

use llm_engine::{Engine, EngineConfig, Event, GenerateRequest, Stop};

/// Shared across tests: model load dominates runtime, so keep it to one.
fn engine() -> Engine {
    Engine::spawn(EngineConfig {
        native_logs: false,
        ..EngineConfig::default()
    })
    .expect("engine failed to start")
}

fn ask(prompt: &str, max_tokens: i32) -> GenerateRequest {
    GenerateRequest {
        max_tokens,
        ..GenerateRequest::user(prompt)
    }
}

/// Drain a generation, returning the text and why it stopped.
async fn drain(generation: &mut llm_engine::Generation) -> (String, Stop) {
    let mut text = String::new();
    while let Some(event) = generation.recv().await {
        match event {
            Event::Token(piece) => text.push_str(&piece),
            Event::Done { stop, .. } => return (text, stop),
            Event::Failed(error) => panic!("generation failed: {error}"),
        }
    }
    panic!("generation ended without a Done event");
}

#[tokio::test(flavor = "multi_thread")]
async fn streams_to_completion() {
    let engine = engine();
    let mut generation = engine.submit(ask("Say hi.", 64)).unwrap();
    let (text, stop) = drain(&mut generation).await;

    assert_eq!(stop, Stop::EndOfGeneration);
    assert!(!text.is_empty(), "no text was produced");
    assert!(
        !text.contains("<think>"),
        "reasoning block leaked into output: {text}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn stops_at_the_token_limit() {
    let engine = engine();
    let mut generation = engine.submit(ask("Write a long essay about the sea.", 8)).unwrap();
    let (_, stop) = drain(&mut generation).await;
    assert_eq!(stop, Stop::Limit);
}

/// Asking for more tokens than the context can hold must end the generation,
/// not fail it: llama.cpp's own error for decoding past the window arrives
/// mid-sentence and says nothing useful.
#[tokio::test(flavor = "multi_thread")]
async fn stops_at_the_edge_of_the_context() {
    let engine = Engine::spawn(EngineConfig {
        n_ctx: 512,
        native_logs: false,
        ..EngineConfig::default()
    })
    .expect("engine failed to start");

    let mut generation = engine
        .submit(ask("Count from 1 upward, one number per line, forever.", 4000))
        .unwrap();
    let (text, stop) = drain(&mut generation).await;

    assert_eq!(stop, Stop::ContextFull);
    assert!(!text.is_empty());
}

/// A prompt too long to decode should say that, rather than failing somewhere
/// inside the first forward pass.
#[tokio::test(flavor = "multi_thread")]
async fn rejects_a_prompt_larger_than_the_context() {
    let engine = Engine::spawn(EngineConfig {
        n_ctx: 64,
        native_logs: false,
        ..EngineConfig::default()
    })
    .expect("engine failed to start");

    let mut generation = engine.submit(ask(&"word ".repeat(200), 16)).unwrap();
    match generation.recv().await.expect("stream ended early") {
        Event::Failed(error) => {
            assert!(error.contains("context"), "unhelpful error: {error}");
        }
        other => panic!("expected a failure, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn explicit_cancel_stops_generation() {
    let engine = engine();
    let mut generation = engine.submit(ask("Count from 1 to 500.", 500)).unwrap();

    let mut seen = 0;
    let stop = loop {
        match generation.recv().await.expect("stream ended early") {
            Event::Token(_) => {
                seen += 1;
                if seen == 3 {
                    generation.cancel();
                }
            }
            Event::Done { stop, tokens, .. } => {
                assert!(tokens < 500, "cancel did not take effect: {tokens} tokens");
                break stop;
            }
            Event::Failed(error) => panic!("generation failed: {error}"),
        }
    };

    assert_eq!(stop, Stop::Cancelled);
}

/// The disconnected-client case: dropping the handle must free the slot, and
/// the engine must go on to serve the next request.
#[tokio::test(flavor = "multi_thread")]
async fn dropping_the_handle_cancels_and_the_engine_recovers() {
    let engine = engine();

    let mut abandoned = engine.submit(ask("Count from 1 to 500.", 500)).unwrap();
    for _ in 0..3 {
        match abandoned.recv().await.expect("stream ended early") {
            Event::Token(_) => {}
            other => panic!("expected a token, got {other:?}"),
        }
    }
    drop(abandoned);

    // The engine must become idle again rather than running the abandoned
    // request to its 500-token limit, which would take far longer than this.
    let mut next = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            if let Ok(generation) = engine.submit(ask("Say hi.", 32)) {
                break generation;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("engine never accepted another request");

    let (text, stop) = tokio::time::timeout(Duration::from_secs(30), drain(&mut next))
        .await
        .expect("follow-up generation stalled");

    assert_eq!(stop, Stop::EndOfGeneration);
    assert!(!text.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn capacity_reports_inflight_work() {
    let engine = engine();
    let idle = engine.capacity();
    assert_eq!(idle.inflight, 0);
    assert_eq!(idle.available_slots, idle.max_inflight);

    let mut generation = engine.submit(ask("Say hi.", 32)).unwrap();
    assert_eq!(engine.capacity().inflight, 1);

    let (_, stop) = drain(&mut generation).await;
    assert_eq!(stop, Stop::EndOfGeneration);
    assert_eq!(engine.capacity().inflight, 0, "slot was not released");
}
