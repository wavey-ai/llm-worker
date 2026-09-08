# llmq

Local LLM inference, and a worker that serves it over HTTP.

```
llmq/            the engine. Owns the model, the context, sampling. No HTTP.
llmq-worker/     the adapter. Claims work from the upload-response ring and
                 answers it in the OpenAI chat-completions format.
examples/serve/  an ingress, for trying it out without a second service.
```

The splits are dependency facts rather than conventions. The engine depends on
nothing that speaks HTTP. The worker reaches a ring over HTTP but never serves
it: no server crate, no listening socket, which is what lets the same code run
against a remote ring and an in-process one. The example is the only place a
port is opened.

## The engine

```rust
let engine = Engine::spawn(EngineConfig::default())?;   // blocks until resident
let mut generation = engine.submit(GenerateRequest::user("Say hi."))?;
while let Some(event) = generation.recv().await {
    match event {
        Event::Token(piece) => print!("{piece}"),
        Event::Done { stop, tokens, .. } => break,
        Event::Failed(error) => break,
    }
}
```

`engine.capacity()` reports slot occupancy in the shape a scheduler's
heartbeat wants. `generation.cancel()` stops the work, and so does dropping
the `Generation` — that is the disconnected-client path, and it needs no
coordination with the transport.

Cancellation lands at the next token boundary, so within ~20ms during
generation, but only after prefill finishes for a long prompt.

There is a CLI over the same interface a worker uses:

```
cargo run -p llmq -- --prompt "Say hi." --max-tokens 64
cargo run -p llmq -- --model models/model.gguf --system "Be brief." --think
```

## The worker

One engine attached to the [upload-response](https://github.com/wavey-ai/web-services)
ring, claiming request lanes and streaming tokens back down response lanes.

The GPU is here; the front door is elsewhere:

```
llmq-worker \
  --model models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --ingress-url https://ingress-a:8443 \
  --ingress-url https://ingress-b:8443
```

## The example

`examples/serve` is a front door of your own: an ingress and a worker in one
process, over an in-process ring, so one binary is enough to try the thing
out. Everything the server crate touches lives there.

```
cargo run -p llmq-serve -- \
  --model models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --tls-cert cert.pem --tls-key key.pem
```

```
curl -sk -N https://localhost:8443/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"stream":true}'
```

### On the wire

`POST /v1/chat/completions` in the OpenAI shape. `stream: true` returns
`text/event-stream` chunks ending in `data: [DONE]`; otherwise one
`chat.completion` JSON body. `GET /health` reports the engine's own capacity
and `GET /v1/models` names the model.

Beyond the standard fields: `think` (or `chat_template_kwargs.enable_thinking`)
lets the model emit its reasoning block. Unknown fields are ignored rather
than rejected.

### Capacity

`--max-inflight` sizes the engine, and the engine's `capacity()` is what the
ring hears in its heartbeat — one number, not two. It is 1 while the engine
is single-slot: a second request then waits in the ring, where a scheduler
can see it, rather than inside the engine, where it cannot.

The example runs the public listener only. A worker on another machine reaches
a ring through its private mutually-authenticated control listener, which the
example does not start.
