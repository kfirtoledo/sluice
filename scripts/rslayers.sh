#!/bin/bash
# H-KERNEL [4]: does the router-split cost SCALE with the number of armed
# layers (=> it IS the op restructuring, and it is a floor of the design), or
# is it a fixed property of the capture mode (=> fixable)?
#
#   SLUICE_RS_LAYERS=N  -> only the first N MoE layers carry router-split's op
#                          structure; the rest keep vLLM's stock fused path.
#   With SLUICE_RS_NOOP=1 the classic streaming hook is a passthrough too, so
#   NOTHING streams in any arm and N is the only variable.
#
# INSTRUMENT CHANGE, forced by entry [3]. `bench serve` against a long-lived
# server drifted ~6 ms (22%) WITHIN one instance, monotonically across
# successive runs -- larger than the effect under test, and it made removing a
# memory copy look 4.2 ms "slower". `bench latency` runs in-process: no API
# server, no network, no cross-run scheduler/cache accumulation, and it
# reports a distribution over 30 iterations instead of 2 samples.
#
#   batch-size 8   : matches c=8 and stays under router-split's smallness
#                    threshold (slots//topk = 48/6 = 8), so the split path is
#                    engaged on every step.
#   input-len 8    : keeps prefill to ~4% of steps so decode dominates.
#                    (Prior entries used 32; this sweep re-anchors everything
#                    on this node with this instrument, vanilla included, so
#                    it is internally consistent. Cross-entry absolute numbers
#                    are NOT comparable.)
#   vanilla twice  : first and last. If the anchors disagree, every delta here
#                    must be quoted with that band.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/rslayers.txt; : > "$RES"

bench_arm () {
  local TAG="$1"; shift
  local LOG=/work/rs_$TAG.log
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  env "$@" timeout 3600 vllm bench latency \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.55 \
    --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE \
    --input-len 8 --output-len 200 --batch-size 8 \
    --num-iters-warmup 10 --num-iters 30 \
    >"$LOG" 2>&1 </dev/null
  local AVG PCT ARM
  AVG=$(grep -iE 'Avg latency' "$LOG" | tail -1 | tr -s ' ')
  PCT=$(grep -iE '^\s*(10|25|50|75|90|99)%' "$LOG" | tr -s ' ' | tr '\n' ' ')
  ARM=$(grep -oE 'router-split layers=[0-9]+' "$LOG" | sort -u | tr '\n' ' ')
  if [ -z "$AVG" ]; then
    echo "$TAG | FAILED" >> "$RES"; tail -12 "$LOG" >> "$RES"; return
  fi
  echo "$TAG | $AVG | pct: $PCT | ARMED: ${ARM:-none}" >> "$RES"
}

SL="SLUICE_SLOTS=48 SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_RS_NOOP=1"

bench_arm vanilla_a IGNORE=1          # anchor: plugin is inert without SLUICE_SLOTS
bench_arm rs0       $SL SLUICE_RS_LAYERS=0
bench_arm rs11      $SL SLUICE_RS_LAYERS=11
bench_arm rs22      $SL SLUICE_RS_LAYERS=22
bench_arm rs99      $SL SLUICE_RS_LAYERS=99
bench_arm vanilla_b IGNORE=1          # drift control: must match vanilla_a
echo "RSLAYERS_DONE" >> "$RES"
