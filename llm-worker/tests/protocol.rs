//! The wire format, tested without a model or a socket.

use llm_worker::protocol::{ChatRequest, RequestDefaults, ResponseWriter, Usage};
use llm_engine::Stop;

fn parse(json: &str) -> ChatRequest {
    ChatRequest::parse(json.as_bytes()).expect("request should parse")
}

/// Every SSE frame in `body`, with the `data: ` prefix stripped.
fn frames(body: &str) -> Vec<&str> {
    body.split("\n\n")
        .filter(|frame| !frame.is_empty())
        .map(|frame| frame.strip_prefix("data: ").expect("frame is an SSE data line"))
        .collect()
}

#[test]
fn reads_a_plain_request() {
    let request = parse(
        r#"{"model":"qwen","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
    );
    assert_eq!(request.model_name(), "qwen");
    assert!(request.stream);

    let generate = request.to_generate_request(RequestDefaults::default());
    assert_eq!(generate.messages.len(), 1);
    assert_eq!(generate.messages[0].role, "user");
    assert_eq!(generate.messages[0].content, "hi");
}

/// Content is a string in the original API and a list of parts in the current
/// one. Both arrive from real clients.
#[test]
fn reads_both_content_shapes() {
    let request = parse(
        r#"{"messages":[
            {"role":"system","content":"be brief"},
            {"role":"user","content":[{"type":"text","text":"one "},{"type":"image_url"},{"type":"text","text":"two"}]}
        ]}"#,
    );
    let generate = request.to_generate_request(RequestDefaults::default());
    assert_eq!(generate.messages[0].content, "be brief");
    assert_eq!(generate.messages[1].content, "one two");
}

#[test]
fn unknown_fields_are_ignored_but_empty_messages_are_not() {
    parse(r#"{"messages":[{"role":"user","content":"hi"}],"presence_penalty":0.5,"user":"someone"}"#);
    assert!(ChatRequest::parse(br#"{"messages":[]}"#).is_err());
    assert!(ChatRequest::parse(b"not json").is_err());
}

#[test]
fn sampling_falls_back_to_the_configured_defaults() {
    let defaults = RequestDefaults {
        max_tokens: 64,
        temperature: 0.7,
        top_p: 0.8,
        seed: 7,
        think: false,
    };
    let generate = parse(r#"{"messages":[{"role":"user","content":"hi"}]}"#)
        .to_generate_request(defaults);
    assert_eq!(generate.max_tokens, 64);
    assert_eq!(generate.temperature, 0.7);
    assert_eq!(generate.seed, 7);

    // max_completion_tokens is the current spelling, and wins.
    let generate = parse(
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":10,"max_completion_tokens":20}"#,
    )
    .to_generate_request(defaults);
    assert_eq!(generate.max_tokens, 20);
}

#[test]
fn thinking_is_accepted_under_either_name() {
    let defaults = RequestDefaults::default();
    let ours = parse(r#"{"messages":[{"role":"user","content":"hi"}],"think":true}"#);
    assert!(ours.to_generate_request(defaults).think);

    let theirs = parse(
        r#"{"messages":[{"role":"user","content":"hi"}],"chat_template_kwargs":{"enable_thinking":true}}"#,
    );
    assert!(theirs.to_generate_request(defaults).think);
}

#[test]
fn a_stream_opens_with_a_role_and_ends_with_a_sentinel() {
    let writer = ResponseWriter::new("chatcmpl-1", "qwen");
    let body = format!(
        "{}{}{}",
        writer.stream_open(),
        writer.stream_token("Hi"),
        writer.stream_close(Stop::EndOfGeneration, None)
    );
    let frames = frames(&body);
    assert_eq!(frames.len(), 4);
    assert_eq!(frames[3], "[DONE]");

    let open: serde_json::Value = serde_json::from_str(frames[0]).unwrap();
    assert_eq!(open["object"], "chat.completion.chunk");
    assert_eq!(open["choices"][0]["delta"]["role"], "assistant");
    assert!(open["choices"][0]["delta"]["content"].is_null());

    let token: serde_json::Value = serde_json::from_str(frames[1]).unwrap();
    assert_eq!(token["choices"][0]["delta"]["content"], "Hi");

    let close: serde_json::Value = serde_json::from_str(frames[2]).unwrap();
    assert_eq!(close["choices"][0]["finish_reason"], "stop");
    assert!(close["usage"].is_null(), "usage is only sent when asked for");
}

#[test]
fn the_token_limit_is_reported_as_length() {
    let writer = ResponseWriter::new("chatcmpl-2", "qwen");
    let body = writer.stream_close(Stop::Limit, Some(Usage::new(5, 8)));
    let close: serde_json::Value = serde_json::from_str(frames(&body)[0]).unwrap();
    assert_eq!(close["choices"][0]["finish_reason"], "length");
    assert_eq!(close["usage"]["prompt_tokens"], 5);
    assert_eq!(close["usage"]["completion_tokens"], 8);
    assert_eq!(close["usage"]["total_tokens"], 13);
}

#[test]
fn a_buffered_completion_carries_the_whole_message() {
    let writer = ResponseWriter::new("chatcmpl-3", "qwen");
    let body = writer.completion("Hello.".to_string(), Stop::EndOfGeneration, Usage::new(4, 2));
    let completion: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(completion["object"], "chat.completion");
    assert_eq!(completion["id"], "chatcmpl-3");
    assert_eq!(completion["choices"][0]["message"]["role"], "assistant");
    assert_eq!(completion["choices"][0]["message"]["content"], "Hello.");
    assert_eq!(completion["choices"][0]["finish_reason"], "stop");
    assert_eq!(completion["usage"]["total_tokens"], 6);
}
