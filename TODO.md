# TODO

What we know, what we guessed, and what is left. Measurements are from an M1
Air (68 GB/s, 16GB) running Qwen3.5-0.8B Q4_K_M, release builds, via
`examples/serve/load.sh`.

## The number to beat

```
step = 34ms + 7.0ms x batch        (fitted over batch 8..32, near-exact)
```

Batch 1 sits off that line at 19ms, below the constant, so llama.cpp takes a
different kernel path once there is more than one token to decode. The 7ms is
the marginal cost of one more sequence and it is what caps aggregate
throughput near **143 tok/s** however many slots exist. We reach 124.6 at 32.

## Open: what is the 7ms?

Batching should be nearly free — the same weights are read once for every
sequence in the batch — and it is not. Each extra sequence costs about half a
full forward pass.

Ruled out by measurement:

- **The engine loop.** Sampling, detokenizing and handing tokens to callers
  are 2% of a step. Zero idle spins between decodes.
- **Context length.** 8 slots at ctx 512 and ctx 4096 are within noise of each
  other (88.3 vs 88.9 tok/s), so it is not attention scanning an allocated
  window.
- **`kv_unified`.** llama.cpp defaults it false, which splits a batch into one
  ubatch per sequence — a good suspect. Forcing it true (confirmed in
  llama.cpp's own log) changed nothing.

Still open, and the leading hypothesis: **the model is a hybrid.** `arch =
qwen35` keeps a KV cache on only 6 of its 24 layers; the other 18 are
linear-attention layers carrying recurrent state per sequence, and
per-sequence state does not amortise across a batch the way shared weights do.
Supporting numbers: 7ms at the ~27 GB/s this machine actually achieves is
~190MB per sequence per step, about 40% of the model; and the same GPU batches
23 tokens of *one* sequence at 0.4ms each during prefill against 8 tokens of
*eight* sequences at 9.9ms each during decode.

**To confirm:** run the same sweep against a dense transformer of similar
size. `unsloth/Qwen2.5-0.5B-Instruct-GGUF` returned 401 — wrong repo name, or
the token in `~/.hf_token` is stale. If a dense model scales cleanly with
batch, the hypothesis holds and the ceiling is a property of this model rather
than of the engine.

## Next

**Chunked prefill.** A joining slot gets its own decode (`Runtime::prefill`),
so every running slot stalls for the length of that prompt. Fine at 25 tokens,
not at 2000. The fix is a token budget per step: put a slice of the prompt in
the same batch as the decodes, `n_ubatch` minus the active slots. This is the
one that makes admission smooth under real prompts.

**Prefix reuse.** Every turn of a conversation re-prefills the whole history —
the chat page resends it all, and the engine tokenizes it all. At ~2370 tok/s
prefill a 2000-token thread wastes ~840ms per turn. With the persistent
context we now have, keep the KV for the common prefix and `kv_cache_seq_rm`
from the point of divergence. Biggest single latency win available.

**Where the queue lives.** With `--target-step-ms` set, requests that are
accepted but not admitted wait *inside* the engine and still count as
`inflight`. Arguably they belong in the ring, where a scheduler can see them.
`capacity()` now reports `decoding` alongside `inflight`, but
`WorkerHeartbeatUpdate` has no field for it, so the ring cannot tell the
difference between busy and merely accepted.

**Control listener.** `llm-worker` — the production binary — cannot attach to
our own `examples/serve` ingress: a worker reaches a ring through its private
mutually-authenticated control listener (`UploadResponseControlRouter`), which
the example does not start. The remote path is wired and compiles but has
never run end to end. Needs `--control-port`, a client CA, and a worker
identity on both sides.

## Fairness

Fixed upstream in `af3f884`: `active_streams()` returned streams in slot
order, and slots come off a stack and are reused newest-first, so a request
parked in a rarely reached slot waited behind every stream that arrived after
it — under sustained load it need never be reached at all. Now sorted by
stream id, which is arrival order.

It made no measurable difference to p50, p95 or throughput, and that is
expected: `load.sh` is closed-loop with uniform requests, so ordering changes
who waits rather than how long the set takes. The fix is insurance against
starvation, not a throughput win.

Still open:

- **Per-tenant fairness.** FIFO is fair between requests, not between callers:
  one client sending 50 requests takes 50 places in the queue. Round-robin
  needs a tenant key off a header, which is a design conversation.
- **Free-slot order.** `allocate_slot` pops a LIFO stack, so slot reuse
  concentrates on low indices. A queue would spread it.
- **An open-loop harness.** `load.sh` cannot demonstrate starvation or measure
  queue wait under sustained overload, which is exactly where the fairness
  work pays. Different tool: arrivals at a fixed rate above capacity,
  measuring worst-case wait rather than percentiles over a closed set.

## Not available here

**PagedAttention.** It is a block-table allocator plus an attention kernel
that gathers KV from non-contiguous blocks. llama.cpp implements neither, and
`llama-cpp-2` exposes only sequence-level memory operations
(`kv_cache_seq_rm`, `seq_cp`, `seq_keep`, `seq_add`), so there is no
block-level handle to build one from. Getting it means changing engine —
vLLM/SGLang, or mistral.rs / candle-vllm in Rust.

The one piece of its value that might be reachable: cells carry a set of
sequence ids, so `kv_cache_seq_cp` should *share* a prefix rather than copy
it — a shared system prompt prefilled once. Worth verifying before relying on
it.

## Smaller

- **Flash attention.** `with_flash_attention_policy` exists and is unmeasured.
  Should matter more as contexts grow.
- **Speculative decoding.** `llama-cpp-2` ships `speculative.rs`. It raises
  single-stream speed, which is the thing batching does not help — but it
  wants a draft model much smaller than the target, so it is for larger models
  than this one.
- **Stall policy.** A slot whose caller stops reading drops out of the batch
  and is abandoned after 30s (`STALL_TIMEOUT`). The mechanism is tested; the
  timeout has never been reached under real backpressure.
- **`n_batch`.** Set to one slot's context so a full-window prefill fits.
  Untuned; `n_ubatch` is left at llama.cpp's 512.

## Measuring, carefully

Three ways this bench has lied to us already, all fixed:

- Wall clock from `date +%s` quantised every aggregate to integer seconds, and
  the "context size matters" reading was that artifact. It now uses curl's own
  millisecond timings.
- A `--log` override suppressed the line a wait loop was grepping for, which
  span a CPU while claiming to measure.
- Debug builds are ~4% slower than release here, which is small enough to
  mistake for a real effect. Always `--release`.

Single runs vary by 20% at the same settings. Take medians of three, and
distrust anything smaller than that.
