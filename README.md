# llm-worker

LLM inference on llama.cpp, served over HTTP through the
[upload-response](https://github.com/wavey-ai/web-services) ring.

## Layout

```
llm-engine/      Inference engine. Loads the model, runs the decode loop.
llm-worker/      Ring adapter and worker binary. Speaks OpenAI chat completions.
examples/serve/  HTTP ingress, an in-process worker, and a chat page.
```

`llm-engine` has no HTTP dependency. `llm-worker` uses HTTP to reach a ring but
does not serve one: no server crate, no listening socket. `examples/serve` is
the only crate that binds a port.

Models are read from the Hugging Face cache, or from a path given with
`--model`. Default is `unsloth/Qwen3.5-0.8B-GGUF`.

## llm-engine

```rust
let engine = Engine::spawn(EngineConfig::default())?;   // blocks until loaded
let mut generation = engine.submit(GenerateRequest::user("Say hi."))?;
while let Some(event) = generation.recv().await {
    match event {
        Event::Token(piece) => print!("{piece}"),
        Event::Done { stop, tokens, .. } => break,
        Event::Failed(error) => break,
    }
}
```

`Engine::spawn` returns once the model is resident. `engine.capacity()` returns
slot occupancy in the same fields as the ring's worker heartbeat.

Generation stops on an end-of-generation token, at `max_tokens`, at the end of
the context window, or on cancellation. `generation.cancel()` cancels; so does
dropping the `Generation`. Cancellation takes effect at the next token
boundary, about 20ms during generation, but not until prefill completes.

CLI:

```
cargo run -p llm-engine -- --prompt "Say hi." --max-tokens 64
cargo run -p llm-engine -- --system "Be brief." --think
```

## llm-worker

Runs one engine and pulls work from one or more ingress services. Claims a
request lane, generates, writes tokens to the matching response lane.

```
llm-worker \
  --ingress-url https://ingress-a:8443 \
  --ingress-url https://ingress-b:8443
```

No listening socket. Workers attach to a ring through its private
mutually-authenticated control listener, which `examples/serve` does not start,
so this binary cannot attach to the example.

## examples/serve

An ingress, a worker and a chat page in one process over an in-process ring.

```
openssl req -x509 -newkey rsa:2048 -nodes -days 1 \
  -keyout key.pem -out cert.pem -subj "/CN=localhost"

cargo run -p llm-serve -- --tls-cert cert.pem --tls-key key.pem
```

Open <https://localhost:8443/>. The certificate is self-signed; accept the
browser warning.

The page is `examples/serve/ui/index.html`, embedded in the binary. No build
step, no dependencies. It streams with `fetch` and a reader over the SSE body.
Stop calls `AbortController.abort()`, which closes the ring's stream and
cancels the generation.

### API

`POST /v1/chat/completions` takes an OpenAI chat completions request. With
`stream: true` the response is `text/event-stream` chunks terminated by
`data: [DONE]`, otherwise a single `chat.completion` JSON body.

`GET /health` returns capacity and model information. `GET /v1/models` returns
the model name.

Two non-standard fields are accepted: `think` and
`chat_template_kwargs.enable_thinking`, either of which lets the model emit its
reasoning block. Unknown fields are ignored.

### load.sh

`examples/serve/load.sh` sends N completions and reports per-request timings.

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

Requests start together unless `ARRIVE=<ms>` is set, which spreads the start
times at random over that many milliseconds. Results are in
[PERFORMANCE.md](PERFORMANCE.md).

### Flags

`--max-inflight` sets the slot count: how many requests decode in one batch.
`--target-step-ms` caps how long a decode step may take instead, and the engine
admits fewer requests to stay under it. Step cost is measured every step and
admission pauses while the measurement is over target. Each sequence produces
one token per step, so the target is also the per-client token rate: 100ms is
10 tokens per second per client. Zero, the default, leaves admission to
`--max-inflight`.

`--ctx-size` is per slot. Eight slots at 4096 allocates a 32768-token context
and the KV memory for it.

`--ring-streams` (`llm-serve`) sets how many requests the ring parks at once.
Beyond that it returns 503.

`capacity()` reports `inflight` (accepted, not finished), `decoding` (in the
current batch), `max_inflight` and `available_slots`. The ring's heartbeat
carries the first, third and fourth. Requests accepted but not yet admitted
count as inflight.

## Other documents

- [PERFORMANCE.md](PERFORMANCE.md) — throughput and latency measurements.
- [TODO.md](TODO.md) — open questions and planned work.
