# Performance

Measurements from an M1 Air (68 GB/s, 16GB) running Qwen3.5-0.8B Q4_K_M,
release builds, via `examples/serve/load.sh`.

Single runs vary by up to 20% at the same settings. Figures below are medians
of three unless stated.

## Batching

| slots | clients | aggregate | time to first byte |
|-------|---------|-----------|--------------------|
| 1     | 8       | 51.8 tok/s | 0.01s to 4.3s, in steps |
| 4     | 8       | 86.5 tok/s | |
| 8     | 8       | 88.9 tok/s | 0.01s for all |
| 8     | 32      | 92.8 tok/s | |
| 16    | 32      | 109.9 tok/s | |
| 32    | 32      | 124.6 tok/s | |

A single client is unaffected by the slot count: 50.3 tok/s with one slot,
51.5 tok/s with eight.

Step time is linear in batch size from batch 8 up:

```
batch   aggregate     step     per sequence
    1   51.8 tok/s    19 ms    19.3 ms
    8   88.9 tok/s    90 ms    11.2 ms
   16  109.9 tok/s   146 ms     9.1 ms
   32  124.6 tok/s   257 ms     8.0 ms

step = 34ms + 7.0ms x batch    →  ceiling ≈ 143 tok/s
```

7.0ms is the marginal cost of one more sequence in the batch. It caps aggregate
throughput near 143 tok/s at any slot count. The 34ms appears only above batch
1: a batch of one takes 19ms, less than that constant, because llama.cpp uses a
different kernel path for a single token.

At 32 clients on 32 slots each client gets about 4 tok/s, against 50 tok/s on
an idle engine.

## Slot count

32 requests arriving at random over four seconds (`ARRIVE=4000`):

| slots | p50    | p95    | spread | aggregate |
|-------|--------|--------|--------|-----------|
| 4     | 6.30s  | 12.76s | 6.5s   | 84.5 tok/s |
| 8     | 6.64s  | 12.30s | 5.7s   | 86.6 tok/s |
| 16    | 6.32s  | 9.96s  | 3.6s   | 102.2 tok/s |
| 32    | 8.68s  | 9.12s  | 0.4s   | 111.2 tok/s |

With fewer slots than concurrent arrivals the surplus requests queue: p50 stays
low, p95 is long. With more slots than arrivals nothing queues and every batch
is as large as demand allows: p95 and throughput improve, p50 gets worse. At 32
slots p50 and p95 are 0.4s apart.

16 slots is the best setting for this load: same p50 as 4 slots, 2.8s off p95,
21% more throughput. 32 slots adds 9% throughput and 38% to p50.

The best number depends on the load. Throughput is paid for in per-request
latency at 7ms per sequence per step.

## --target-step-ms

32 slots available, 32 clients arriving at once:

| target | batch used | measured step |
|--------|------------|---------------|
| off    | 32         | 207ms |
| 60ms   | 7          | 64ms  |
| 100ms  | 10         | 77ms  |
| 200ms  | 18         | 132ms |

32 requests over four seconds:

| | stream rate | p50 | p95 | aggregate |
|--|-------------|-----|-----|-----------|
| no target   | 4.8 tok/s | 8.60s | 9.11s  | 113.2 tok/s |
| target 60ms | 6.0 tok/s | 6.24s | 11.22s | 86.5 tok/s |

Tokens arrive 25% faster and p50 is 27% lower, at the cost of 23% on p95 and
24% of throughput.

## Where step time goes

`llm_engine=debug` logs one line per step:

```
step slots=8 enqueue_us=1043 gpu_us=79214
```

Metal executes `decode` asynchronously. The call returns in about 1ms and the
first read of the logits blocks until the GPU finishes, so `gpu_us` is the cost
of the step: 79ms at batch 8, about 19ms at batch 1. Sampling, detokenizing and
delivering tokens are 2% of a step.

Ruled out as the cause of the 7ms per sequence:

- The engine loop. It is that 2%, and there are no idle spins between decodes.
- Context length. 8 slots at ctx 512 and ctx 4096 measure 88.3 and 88.9 tok/s.
- `kv_unified`. llama.cpp defaults it to false, which splits a batch into one
  ubatch per sequence. Forcing it true, confirmed in llama.cpp's log, changed
  nothing.

Probable cause, unconfirmed: `arch = qwen35` is a hybrid. 6 of 24 layers keep a
KV cache; the other 18 are linear-attention layers holding recurrent state per
sequence, which does not amortise across a batch the way shared weights do.
7ms at the ~27 GB/s this machine achieves is ~190MB per sequence per step,
about 40% of the model. The same GPU processes 23 tokens of one sequence at
0.4ms each during prefill and 8 tokens of eight sequences at 9.9ms each during
decode. Confirming this needs a dense model of similar size. See `TODO.md`.

## Ring fairness

`upload-response` af3f884 changed `active_streams()` to return streams in
arrival order. Before that it returned them in slot order, and slots are
allocated from a stack and reused newest-first, so a request in a rarely
reached slot waited behind every stream that arrived after it.

The change made no measurable difference to p50, p95 or throughput.
`load.sh` is closed-loop with uniform requests, so ordering changes which
request waits, not how long the set takes. Demonstrating the difference needs
an open-loop harness: arrivals at a fixed rate above capacity, measuring
worst-case wait.

## Measurement notes

Errors found in this benchmark and fixed:

- Wall clock from `date +%s` quantised aggregate figures to integer seconds.
  An apparent effect of context size on throughput was that artifact. Now uses
  curl's millisecond timings.
- A `--log` override suppressed the line a startup wait loop was grepping for,
  which spun a CPU during a measurement.
- Debug builds are about 4% slower than release here, small enough to mistake
  for a real effect.
