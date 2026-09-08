# llm-worker

Local LLM inference, and a worker that serves it over HTTP.

```
llm-engine/      the engine. Owns the model, the context, sampling. No HTTP.
llm-worker/      the adapter. Claims work from the upload-response ring and
                 answers it in the OpenAI chat-completions format.
examples/serve/  an ingress, for trying it out without a second service.
```

The splits are dependency facts rather than conventions. The engine depends on
nothing that speaks HTTP. The worker reaches a ring over HTTP but never serves
it: no server crate, no listening socket, which is what lets the same code run
against a remote ring and an in-process one. The example is the only place a
port is opened.

The model comes from the Hugging Face cache, downloaded on first use and
shared with everything else on the machine that pulls from it. `--model`
takes a GGUF file directly if you would rather not.

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
cargo run -p llm-engine -- --system "Be brief." --think
```

## The worker

One engine attached to the [upload-response](https://github.com/wavey-ai/web-services)
ring, claiming request lanes and streaming tokens back down response lanes.

The GPU is here; the front door is elsewhere:

```
llm-worker \
  --ingress-url https://ingress-a:8443 \
  --ingress-url https://ingress-b:8443
```

## The example

`examples/serve` is a front door of your own: an ingress, a worker, and a chat
page in one process, over an in-process ring, so one binary is enough to try
the thing out. Everything the server crate touches lives there.

```
cargo run -p llm-serve -- --tls-cert cert.pem --tls-key key.pem
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

Time to first byte is the tell. With one slot it climbs in one-generation
steps, because each request is waiting for the slot rather than for the GPU.

### What batching buys, measured

An M1 Air, Qwen3.5-0.8B Q4_K_M, eight clients at once:

| slots | aggregate | time to first byte |
|-------|-----------|--------------------|
| 1     | 51.8 tok/s | 0.01s → 4.3s, in steps |
| 2     | 57.4 tok/s | |
| 4     | 86.5 tok/s | |
| 8     | 88.9 tok/s | ~0.01s, all of them |
| 16 slots, 16 clients | 109.8 tok/s | |

A lone client is unaffected by the slot count: 50.3 tok/s at one slot, 51.5 at
eight. Nobody pays for capacity they are not using.

The latency win is unambiguous — everyone starts at once instead of queueing.
The throughput win is real but smaller than the theory says it should be, and
`llm_engine=debug` says where it goes:

```
step slots=8 enqueue_us=1043 gpu_us=79214
```

Metal runs `decode` asynchronously, so the enqueue returns in 1ms and the
first read of the logits waits for the GPU. That wait is 79ms for a batch of
eight, against ~19ms for a batch of one — so each extra sequence costs about
half a full forward pass instead of nearly nothing. Everything outside the GPU
— sampling, detokenizing, handing tokens to callers — is 2% of a step.

Two suspects ruled out and one still open. It is not the engine loop, which is
that 2%. It is not `kv_unified`, which llama.cpp defaults to false: forcing it
true changed nothing. What remains is the model. `arch = qwen35` is a hybrid —
only 6 of its 24 layers keep a KV cache, the other 18 are linear-attention
layers carrying recurrent state per sequence, and per-sequence state does not
amortise across a batch the way shared weights do. The same GPU batches 23
tokens of one sequence at 0.4ms each during prefill and 8 tokens of eight
sequences at 9.9ms each during decode. Confirming that needs a dense
transformer of similar size to compare against.

### Capacity

`--max-inflight` sizes the engine, and the engine's `capacity()` is what the
ring hears in its heartbeat — one number, not two. Requests beyond that queue
inside the engine and still count as occupancy, so a scheduler reading
`available_slots` is never told there is room that does not exist.

`--ctx-size` is **per slot**. Eight slots at 4096 is a 32768-token context,
and the KV memory to match.

The example runs the public listener only. A worker on another machine reaches
a ring through its private mutually-authenticated control listener, which the
example does not start.
