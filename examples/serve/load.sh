#!/usr/bin/env bash
#
# Fire N completions at an ingress at once and report what came back.
#
#   ./examples/serve/load.sh              4 requests at https://localhost:8443
#   N=16 ./examples/serve/load.sh         16 at once
#   HOST=https://box:8443 ./examples/serve/load.sh
#
# The engine is single-slot, so this measures a queue, not a batch: total time
# grows with N while each request's own generation stays about as fast. Time to
# first byte is the tell — it is mostly time spent waiting for the slot.

set -euo pipefail

HOST=${HOST:-https://localhost:8443}
N=${N:-4}
MAX_TOKENS=${MAX_TOKENS:-64}
PROMPT=${PROMPT:-"In one sentence, what is a vector database?"}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

if ! curl -sk --max-time 5 "$HOST/health" -o "$work/health"; then
  echo "no ingress at $HOST" >&2
  exit 1
fi
echo "ingress: $(tr -d '\n' < "$work/health")"
echo "sending $N requests, max_tokens=$MAX_TOKENS"
echo

# Sample capacity while the requests run, so the engine's own view of
# occupancy is visible next to the client's.
(
  while :; do
    curl -sk --max-time 2 "$HOST/health" 2>/dev/null |
      grep -o '"inflight":[0-9]*' | cut -d: -f2 >> "$work/inflight"
  done
) &
sampler=$!
disown "$sampler" 2>/dev/null || true   # so killing it later stays quiet

started=$(date +%s)

pids=()
for i in $(seq 1 "$N"); do
  (
    curl -sk -N -X POST "$HOST/v1/chat/completions" \
      -H 'content-type: application/json' \
      -w '\nttfb=%{time_starttransfer} total=%{time_total} status=%{http_code}\n' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
           \"stream\":true,\"max_tokens\":$MAX_TOKENS,
           \"stream_options\":{\"include_usage\":true}}" \
      > "$work/$i.out" 2>&1
  ) &
  pids+=($!)
done
# Only the requests. A bare `wait` would also wait for the sampler, which
# never returns on its own.
wait "${pids[@]}"

elapsed=$(( $(date +%s) - started ))
kill "$sampler" 2>/dev/null || true

printf '%3s  %7s  %7s  %7s  %7s  %s\n' "#" "ttfb" "total" "tokens" "tok/s" "finish"
total_tokens=0
for i in $(seq 1 "$N"); do
  out="$work/$i.out"
  status=$(grep -o 'status=[0-9]*' "$out" | cut -d= -f2)
  ttfb=$(grep -o 'ttfb=[0-9.]*' "$out" | cut -d= -f2)
  total=$(grep -o 'total=[0-9.]*' "$out" | cut -d= -f2)
  tokens=$(grep -o '"completion_tokens":[0-9]*' "$out" | tail -1 | cut -d: -f2)
  finish=$(grep -o '"finish_reason":"[a-z]*"' "$out" | tail -1 | cut -d'"' -f4)

  if [ "$status" != "200" ] || [ -z "${tokens:-}" ]; then
    # 503 means the ring refused the stream: more requests at once than it
    # has slots to park them in.
    printf '%3s  %7s  %7s  %7s  %7s  %s\n' "$i" "${ttfb:--}" "${total:--}" "-" "-" "http ${status:-?}"
    continue
  fi

  # Generation time is what is left after the wait for a slot.
  rate=$(awk -v t="$tokens" -v a="$total" -v b="$ttfb" 'BEGIN { d = a - b; printf "%.1f", (d > 0 ? t / d : 0) }')
  total_tokens=$(( total_tokens + tokens ))
  printf '%3s  %7s  %7s  %7s  %7s  %s\n' "$i" "$ttfb" "$total" "$tokens" "$rate" "$finish"
done

echo
echo "wall ${elapsed}s · $total_tokens tokens · $(awk -v t="$total_tokens" -v s="$elapsed" 'BEGIN { printf "%.1f", (s > 0 ? t / s : t) }') tok/s across all requests"
if [ -s "$work/inflight" ]; then
  echo "engine inflight while running: max $(sort -n "$work/inflight" | tail -1) of $(grep -o '"max_inflight":[0-9]*' "$work/health" | cut -d: -f2)"
fi
