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

| slots | clients | aggregate | time to first byte |
|-------|---------|-----------|--------------------|
| 1     | 8       | 51.8 tok/s | 0.01s → 4.3s, in steps |
| 4     | 8       | 86.5 tok/s | |
| 8     | 8       | 88.9 tok/s | ~0.01s, all of them |
| 8     | 32      | 92.8 tok/s | |
| 16    | 32      | 109.9 tok/s | |
| 32    | 32      | 124.6 tok/s | |

A lone client is unaffected by the slot count: 50.3 tok/s at one slot, 51.5 at
eight. Nobody pays for capacity they are not using.

Step time is linear in the batch, and the line is a good fit from 8 slots up:

```
batch   aggregate     step     per sequence
    1   51.8 tok/s    19 ms    19.3 ms
    8   88.9 tok/s    90 ms    11.2 ms
   16  109.9 tok/s   146 ms     9.1 ms
   32  124.6 tok/s   257 ms     8.0 ms

step = 34ms + 7.0ms x batch    →  ceiling ≈ 143 tok/s
```

Two numbers in that line matter. The 7ms per sequence is what stops batching
paying off the way it should; it is the marginal cost of one more sequence in
the batch, and it puts a ceiling near 143 tok/s however many slots there are.
The 34ms is a fixed cost that appears only above batch 1 — a batch of one
takes 19ms, less than the constant — which is llama.cpp taking a different
kernel path once there is more than one token to decode.

Aggregate throughput is not the only thing that moves. At 32 clients on 32
slots each one sees about 4 tok/s, against 50 on its own. That is the trade
being made: everyone starts immediately and nobody waits in line, but a busy
engine is slower for each of them than an idle one.

### Choosing the slot count

All-at-once is a worst case, not a workload. `ARRIVE` spreads the start times,
which is what decides how full the batches actually get:

```
N=32 ARRIVE=4000 MAX_TOKENS=48 ./examples/serve/load.sh
```

32 requests arriving at random over four seconds, median of three runs:

| slots | p50    | p95    | spread | aggregate |
|-------|--------|--------|--------|-----------|
| 4     | 6.30s  | 12.76s | 6.5s   | 84.5 tok/s |
| 8     | 6.64s  | 12.30s | 5.7s   | 86.6 tok/s |
| 16    | 6.32s  | 9.96s  | 3.6s   | 102.2 tok/s |
| 32    | 8.68s  | 9.12s  | 0.4s   | 111.2 tok/s |

Below the arrival rate, slots are a queue: the median is fine because most
requests find a free slot, and the tail is bad because the unlucky ones wait.
Above it, nothing queues and every batch is as full as demand allows: the tail
and the total improve, and everybody's median gets worse together. At 32 slots
p50 and p95 are 0.4s apart — perfectly fair, uniformly slower.

Sixteen is the knee here: p50 no worse than at four slots, p95 nearly three
seconds better, 21% more throughput. Thirty-two buys another 9% of throughput
for 38% on the median.

The knee is a property of the load, not of the engine, so it moves. What does
not move is the shape: **capacity is bought with individual latency, and the
exchange rate gets worse as the batch grows** — 7ms per sequence, every step,
for every sequence in it.

The latency win is unambiguous — everyone starts at once instead of queueing.
The throughput win is real but smaller than the theory says it should be, and
`llm_engine=debug` says where it goes:

```
step slots=8 enqueue_us=1043 gpu_us=79214
```

(`enqueue_us` is the call; `gpu_us` is the wait for it to actually happen.)

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
