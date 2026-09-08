# llm-worker

Local LLM inference, and a worker that serves it over HTTP.

```
llm-engine/            the engine. Owns the model, the context, sampling. No HTTP.
llm-worker/     the adapter. Claims work from the upload-response ring and
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
cargo run -p llm-engine -- --prompt "Say hi." --max-tokens 64
cargo run -p llm-engine -- --model models/model.gguf --system "Be brief." --think
```

## The worker

One engine attached to the [upload-response](https://github.com/wavey-ai/web-services)
ring, claiming request lanes and streaming tokens back down response lanes.

The GPU is here; the front door is elsewhere:

```
llm-worker \
  --model models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --ingress-url https://ingress-a:8443 \
  --ingress-url https://ingress-b:8443
```

## The example

`examples/serve` is a front door of your own: an ingress, a worker, and a chat
page in one process, over an in-process ring, so one binary is enough to try
the thing out. Everything the server crate touches lives there.

```
cargo run -p llm-serve -- \
  --model models/Qwen3.5-0.8B-Q4_K_M.gguf \
  --tls-cert cert.pem --tls-key key.pem
```

Then open <https://localhost:8443/> — a self-signed certificate means the
browser will want convincing first. The page is `examples/serve/ui/index.html`,
embedded in the binary: no build step, no dependencies, streaming by
`fetch` and a reader over the SSE body. Its Stop button aborts the request,
which drops the connection, which closes the ring's stream, which cancels the
generation — the whole path in one click.

Generate a certificate to test with:

```
openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
  -keyout key.pem -out cert.pem -subj "/CN=localhost"
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

### Load

`examples/serve/load.sh` fires N completions at an ingress at once and reports
what came back.

```
N=8 MAX_TOKENS=32 ./examples/serve/load.sh

  #     ttfb    total   tokens    tok/s  finish
  1  0.013832  0.631942       32     51.8  length
  2  1.266202  1.869136       32     53.1  length
  3  0.642347  1.252220       32     52.5  length
  ...
  8  4.329184  4.941378       32     52.3  length

wall 5s · 256 tokens · 51.2 tok/s across all requests
engine inflight while running: max 1 of 1
```

Read it as the baseline it is. Time to first byte climbs in one-generation
steps because each request waits for the slot; each generation runs at full
speed once it has it; and aggregate throughput is the same 52 tok/s a single
stream gets. The GPU is doing one sequence's work no matter how many clients
are waiting, which is what the multi-slot loop is for.

### Capacity

`--max-inflight` sizes the engine, and the engine's `capacity()` is what the
ring hears in its heartbeat — one number, not two. It is 1 while the engine
is single-slot: a second request then waits in the ring, where a scheduler
can see it, rather than inside the engine, where it cannot.

The example runs the public listener only. A worker on another machine reaches
a ring through its private mutually-authenticated control listener, which the
example does not start.
