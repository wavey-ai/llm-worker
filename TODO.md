# TODO

## Confirm the per-sequence cost

Step time is `34ms + 7.0ms x batch`. The 7ms caps aggregate throughput near
143 tok/s at any slot count. Measurements and ruled-out causes are in
[PERFORMANCE.md](PERFORMANCE.md).

Probable cause is the hybrid architecture of Qwen3.5. To confirm, run the same
sweep against a dense transformer of similar size. `unsloth/Qwen2.5-0.5B-
Instruct-GGUF` returned 401: either the wrong repo name or a stale token in
`~/.hf_token`.

## Chunked prefill

A joining slot gets its own decode in `Runtime::prefill`, so every running slot
stalls for the length of that prompt. At 25 tokens that is 10ms. At 2000 it is
most of a second.

Fix: a token budget per step. Put a slice of the prompt in the same batch as
the decode tokens, sized `n_ubatch` minus the active slots.

## Prefix reuse

Every turn of a conversation re-prefills the whole history. At ~2370 tok/s
prefill, a 2000-token thread costs ~840ms per turn.

Fix: keep the KV for the common prefix and `kv_cache_seq_rm` from the point of
divergence. The persistent context needed for this already exists.

## Queue location

With `--target-step-ms` set, requests waiting for admission wait inside the
engine and count as `inflight`. In the ring a scheduler could see them.
`capacity()` reports `decoding` separately, and `WorkerHeartbeatUpdate` carries
`inflight` alone, so the ring reads waiting and decoding as one number.

## Control listener

Workers reach a ring through its private mutually-authenticated control
listener (`UploadResponseControlRouter`). `examples/serve` starts the public
listener alone, so `llm-worker` needs a ring started elsewhere. The remote path
compiles and has run against an unreachable ingress, which exercises discovery,
heartbeat and claim polling.

Needs `--control-port`, a client CA, and a worker identity on both sides.

## Fairness

Fixed upstream in `af3f884`: streams are handed to workers in arrival order
rather than slot order. Figures on this benchmark matched the previous ones;
the change bounds waiting time under sustained load.

Open:

- Per-tenant fairness. FIFO orders requests, so one client sending 50 requests
  takes 50 places in the queue. Round-robin needs a tenant key from a header.
- `allocate_slot` pops a LIFO stack, so slot reuse concentrates on low indices.
  A queue would spread it.
- An open-loop load harness. `load.sh` is closed-loop; measuring queue wait
  under sustained overload needs arrivals at a fixed rate above capacity.

## PagedAttention

Requires a block-table allocator and an attention kernel that gathers KV from
non-contiguous blocks. `llama-cpp-2` exposes sequence-level memory operations
(`kv_cache_seq_rm`, `seq_cp`, `seq_keep`, `seq_add`), which is the whole of the
available interface. Getting PagedAttention means a different backend: vLLM,
SGLang, mistral.rs or candle-vllm.

Possibly reachable: cells carry a set of sequence ids, so `kv_cache_seq_cp` may
share a prefix rather than copy it, which would allow a shared system prompt to
be prefilled once. Unverified.

## Smaller

- `with_flash_attention_policy` exists and is unmeasured.
- `llama-cpp-2` ships `speculative.rs`. Speculative decoding raises
  single-stream speed, which batching leaves flat. It needs a draft model much
  smaller than the target.
- A slot whose caller stops reading drops out of the batch and is abandoned
  after `STALL_TIMEOUT` (30s). The mechanism has a test. The timeout itself has
  fired only in tests.
- `n_batch` is set to one slot's context so a full-window prefill fits.
  Untuned. `n_ubatch` is left at llama.cpp's default of 512.
