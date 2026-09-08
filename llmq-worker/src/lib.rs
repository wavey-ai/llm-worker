//! llmq on the upload-response ring.
//!
//! [`worker`] is the engine's adapter: it claims a request lane, generates,
//! and writes tokens to the response lane. [`protocol`] is the only part that
//! knows the chat-completions wire format.
//!
//! Neither of them serves HTTP. The ring is reached over it, but nothing here
//! listens on a socket or depends on a server, which is what lets the same
//! code run against an in-process ring and a remote one. An ingress that does
//! listen is in `examples/serve`.

pub mod protocol;
pub mod worker;
