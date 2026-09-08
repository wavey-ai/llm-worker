#!/usr/bin/env bash
#
# Fire N completions at an ingress at once and report what came back.
#
#   ./examples/serve/load.sh              4 requests at https://localhost:8443
#   N=16 ./examples/serve/load.sh         16 at once
#   N=32 ARRIVE=4000 ./examples/serve/load.sh   32 spread over 4 seconds
#   HOST=https://box:8443 ./examples/serve/load.sh
#
# ARRIVE spreads the start times at random over that many milliseconds. All at
# once is a useful worst case but not a realistic one: real requests turn up
# while others are already running, which is what decides how full the batch
# actually gets.

set -euo pipefail

HOST=${HOST:-https://localhost:8443}
N=${N:-4}
ARRIVE=${ARRIVE:-0}
MAX_TOKENS=${MAX_TOKENS:-64}
PROMPT=${PROMPT:-"In one sentence, what is a vector database?"}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

if ! curl -sk --max-time 5 "$HOST/health" -o "$work/health"; then
  echo "no ingress at $HOST" >&2
  exit 1
fi
echo "ingress: $(tr -d '\n' < "$work/health")"
if [ "$ARRIVE" -gt 0 ]; then
  echo "sending $N requests over ${ARRIVE}ms, max_tokens=$MAX_TOKENS"
else
  echo "sending $N requests at once, max_tokens=$MAX_TOKENS"
fi
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

pids=()
for i in $(seq 1 "$N"); do
  (
    if [ "$ARRIVE" -gt 0 ]; then
      delay=$(awk -v r="$RANDOM" -v a="$ARRIVE" 'BEGIN { printf "%.3f", (r % a) / 1000 }')
      sleep "$delay"
      echo "delay=$delay" >> "$work/$i.out"
    fi
    curl -sk -N -X POST "$HOST/v1/chat/completions" \
      -H 'content-type: application/json' \
      -w '\nttfb=%{time_starttransfer} total=%{time_total} status=%{http_code}\n' \
      -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],
           \"stream\":true,\"max_tokens\":$MAX_TOKENS,
           \"stream_options\":{\"include_usage\":true}}" \
      >> "$work/$i.out" 2>&1
  ) &
  pids+=($!)
done
# Only the requests. A bare `wait` would also wait for the sampler, which
# never returns on its own.
wait "${pids[@]}"

kill "$sampler" 2>/dev/null || true

printf '%3s  %7s  %7s  %7s  %7s  %7s  %s\n' "#" "start" "ttfb" "total" "tokens" "tok/s" "finish"
total_tokens=0
# The requests all start within a few milliseconds of each other, so the
# slowest one's own clock is the wall time — and it has millisecond
# resolution, which `date +%s` does not.
slowest=0
for i in $(seq 1 "$N"); do
  out="$work/$i.out"
  status=$(grep -o 'status=[0-9]*' "$out" | cut -d= -f2)
  ttfb=$(grep -o 'ttfb=[0-9.]*' "$out" | cut -d= -f2)
  total=$(grep -o 'total=[0-9.]*' "$out" | cut -d= -f2)
  tokens=$(grep -o '"completion_tokens":[0-9]*' "$out" | tail -1 | cut -d: -f2)
  finish=$(grep -o '"finish_reason":"[a-z]*"' "$out" | tail -1 | cut -d'"' -f4)

  start=$(grep -o 'delay=[0-9.]*' "$out" | cut -d= -f2)
  start=${start:-0.000}

  if [ "$status" != "200" ] || [ -z "${tokens:-}" ]; then
    # 503 means the ring refused the stream: more requests at once than it
    # has slots to park them in.
    printf '%3s  %7s  %7s  %7s  %7s  %7s  %s\n' "$i" "$start" "${ttfb:--}" "${total:--}" "-" "-" "http ${status:-?}"
    continue
  fi

  # Generation time is what is left after the wait for a slot.
  rate=$(awk -v t="$tokens" -v a="$total" -v b="$ttfb" 'BEGIN { d = a - b; printf "%.1f", (d > 0 ? t / d : 0) }')
  total_tokens=$(( total_tokens + tokens ))
  # Wall clock has to include the wait before a late arrival even started.
  finished=$(awk -v s="$start" -v t="$total" 'BEGIN { print s + t }')
  slowest=$(awk -v a="$slowest" -v b="$finished" 'BEGIN { print (b > a ? b : a) }')
  echo "$total" >> "$work/completions"
  printf '%3s  %7s  %7s  %7s  %7s  %7s  %s\n' "$i" "$start" "$ttfb" "$total" "$tokens" "$rate" "$finish"
done

echo
# What one caller waited, not what the fleet achieved: the two move in
# opposite directions as the batch grows, which is the whole trade.
if [ -s "$work/completions" ]; then
  sort -n "$work/completions" -o "$work/completions"
  count=$(wc -l < "$work/completions" | tr -d ' ')
  p50=$(awk -v n="$count" 'NR == int((n + 1) / 2) { printf "%.2f", $1 }' "$work/completions")
  p95=$(awk -v n="$count" 'NR == int(n * 0.95 + 0.5) || NR == n { p = $1 } END { printf "%.2f", p }' "$work/completions")
  slow=$(tail -1 "$work/completions" | awk '{ printf "%.2f", $1 }')
  echo "per request   p50 ${p50}s · p95 ${p95}s · slowest ${slow}s"
fi
echo "aggregate     $total_tokens tokens in $(awk -v s="$slowest" 'BEGIN { printf "%.2f", s }')s · $(awk -v t="$total_tokens" -v s="$slowest" 'BEGIN { printf "%.1f", (s > 0 ? t / s : 0) }') tok/s"
if [ -s "$work/inflight" ]; then
  echo "engine inflight while running: max $(sort -n "$work/inflight" | tail -1) of $(grep -o '"max_inflight":[0-9]*' "$work/health" | cut -d: -f2)"
fi
