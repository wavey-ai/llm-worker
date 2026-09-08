//! The OpenAI chat-completions wire format, in and out.
//!
//! This is the only place that knows what a client sends or expects. The
//! worker deals in [`GenerateRequest`] and [`Event`] on one side and opaque
//! byte frames on the other; translation lives here so both can be tested
//! without a model or a socket.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use llm_engine::{GenerateRequest, Message, Stop};
use serde::{Deserialize, Serialize};

/// Reported back as the model name when a request does not name one.
pub const DEFAULT_MODEL: &str = "llm";

pub const SSE_CONTENT_TYPE: &str = "text/event-stream; charset=utf-8";
pub const JSON_CONTENT_TYPE: &str = "application/json";

/// A chat-completions request. Unknown fields are ignored rather than
/// rejected: clients send plenty of parameters this engine has no opinion
/// about, and failing on them would be hostile for no gain.
#[derive(Debug, Clone, Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
    #[serde(default)]
    pub max_tokens: Option<i32>,
    /// The current spelling of `max_tokens`; it wins when both are present.
    #[serde(default)]
    pub max_completion_tokens: Option<i32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u32>,
    /// Our own extension: let the model emit its reasoning block.
    #[serde(default)]
    pub think: Option<bool>,
    /// The spelling vLLM and llama-server use for the same thing.
    #[serde(default)]
    pub chat_template_kwargs: Option<ChatTemplateKwargs>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatTemplateKwargs {
    #[serde(default)]
    pub enable_thinking: Option<bool>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default)]
    pub content: Option<Content>,
}

/// Content is a string in the original API and a list of typed parts in the
/// current one. Both are in the wild, so accept both.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub text: Option<String>,
}

impl Content {
    /// The text of this content, with non-text parts dropped. An image part
    /// in a text-only engine is a silent omission either way; dropping it
    /// keeps the surrounding text usable.
    pub fn text(&self) -> String {
        match self {
            Self::Text(text) => text.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter(|part| part.kind == "text")
                .filter_map(|part| part.text.as_deref())
                .collect::<Vec<_>>()
                .join(""),
        }
    }
}

/// Defaults for the fields a request may leave out.
#[derive(Debug, Clone, Copy)]
pub struct RequestDefaults {
    pub max_tokens: i32,
    pub temperature: f32,
    pub top_p: f32,
    pub seed: u32,
    pub think: bool,
}

impl Default for RequestDefaults {
    fn default() -> Self {
        let request = GenerateRequest::default();
        Self {
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            seed: request.seed,
            think: request.think,
        }
    }
}

impl ChatRequest {
    pub fn parse(body: &[u8]) -> Result<Self> {
        let request: Self = serde_json::from_slice(body)?;
        if request.messages.is_empty() {
            bail!("messages must not be empty");
        }
        Ok(request)
    }

    pub fn model_name(&self) -> &str {
        self.model.as_deref().unwrap_or(DEFAULT_MODEL)
    }

    pub fn include_usage(&self) -> bool {
        self.stream_options
            .is_some_and(|options| options.include_usage)
    }

    pub fn to_generate_request(&self, defaults: RequestDefaults) -> GenerateRequest {
        let think = self
            .think
            .or_else(|| {
                self.chat_template_kwargs
                    .as_ref()
                    .and_then(|kwargs| kwargs.enable_thinking)
            })
            .unwrap_or(defaults.think);

        GenerateRequest {
            messages: self
                .messages
                .iter()
                .map(|message| Message {
                    role: message.role.clone(),
                    content: message.content.as_ref().map(Content::text).unwrap_or_default(),
                })
                .collect(),
            max_tokens: self
                .max_completion_tokens
                .or(self.max_tokens)
                .unwrap_or(defaults.max_tokens),
            temperature: self.temperature.unwrap_or(defaults.temperature),
            top_p: self.top_p.unwrap_or(defaults.top_p),
            seed: self.seed.unwrap_or(defaults.seed),
            think,
        }
    }
}

/// How a stop reason renders on the wire. Cancellation has no OpenAI
/// spelling; the client that cancelled is usually gone, and one that isn't
/// gets the same "stopped early" it would get from a stop sequence.
pub fn finish_reason(stop: Stop) -> &'static str {
    match stop {
        Stop::EndOfGeneration => "stop",
        Stop::Limit => "length",
        Stop::Cancelled => "stop",
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

impl Usage {
    pub fn new(prompt_tokens: u32, completion_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ChunkChoice {
    index: u32,
    delta: Delta,
    finish_reason: Option<&'static str>,
}

#[derive(Debug, Clone, Serialize)]
struct Chunk<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize)]
struct CompletionMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Clone, Serialize)]
struct CompletionChoice {
    index: u32,
    message: CompletionMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Clone, Serialize)]
struct Completion<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<CompletionChoice>,
    usage: Usage,
}

/// Renders one response: an id and model fixed at the start, then a frame per
/// event. Streaming and buffered share it so the two shapes cannot drift.
pub struct ResponseWriter {
    id: String,
    model: String,
    created: u64,
}

impl ResponseWriter {
    pub fn new(id: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            model: model.into(),
            created: unix_seconds(),
        }
    }

    pub fn id(&self) -> &str {
        &self.id
    }

    /// The opening frame. Carries the assistant role and no content, which is
    /// what clients expect to see before any text arrives.
    pub fn stream_open(&self) -> String {
        self.sse(&Chunk {
            id: &self.id,
            object: "chat.completion.chunk",
            created: self.created,
            model: &self.model,
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: Some("assistant"), content: None },
                finish_reason: None,
            }],
            usage: None,
        })
    }

    pub fn stream_token(&self, text: &str) -> String {
        self.sse(&Chunk {
            id: &self.id,
            object: "chat.completion.chunk",
            created: self.created,
            model: &self.model,
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: Some(text.to_string()) },
                finish_reason: None,
            }],
            usage: None,
        })
    }

    /// The closing frames: an empty delta carrying the finish reason,
    /// optionally usage, then the sentinel that ends the SSE stream.
    pub fn stream_close(&self, stop: Stop, usage: Option<Usage>) -> String {
        let mut out = self.sse(&Chunk {
            id: &self.id,
            object: "chat.completion.chunk",
            created: self.created,
            model: &self.model,
            choices: vec![ChunkChoice {
                index: 0,
                delta: Delta { role: None, content: None },
                finish_reason: Some(finish_reason(stop)),
            }],
            usage,
        });
        out.push_str("data: [DONE]\n\n");
        out
    }

    pub fn completion(&self, text: String, stop: Stop, usage: Usage) -> String {
        let completion = Completion {
            id: &self.id,
            object: "chat.completion",
            created: self.created,
            model: &self.model,
            choices: vec![CompletionChoice {
                index: 0,
                message: CompletionMessage { role: "assistant", content: text },
                finish_reason: finish_reason(stop),
            }],
            usage,
        };
        serde_json::to_string(&completion).expect("completion is serializable")
    }

    fn sse<T: Serialize>(&self, frame: &T) -> String {
        let json = serde_json::to_string(frame).expect("frame is serializable");
        format!("data: {json}\n\n")
    }
}

/// An error in the shape OpenAI clients parse, for the paths where nothing
/// has been written yet and a status code is still ours to choose.
pub fn error_body(message: &str, kind: &str) -> String {
    serde_json::json!({
        "error": { "message": message, "type": kind, "param": null, "code": null }
    })
    .to_string()
}

/// A mid-stream failure. The status line is long gone by then, so the error
/// travels as a frame and the stream still terminates properly.
pub fn error_frame(message: &str) -> String {
    format!("data: {}\n\ndata: [DONE]\n\n", error_body(message, "server_error"))
}

fn unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}
