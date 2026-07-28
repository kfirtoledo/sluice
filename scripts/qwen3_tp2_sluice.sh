#!/bin/bash
# pds-perf [3d]: the Qwen3 Sluice arm at TP=2, with gpu-util lowered.
#
# WHY IT OOMed AT 0.90 (twice, and my first explanation was wrong):
# Sluice's peak is NOT "weights + slot cache". It is WEIGHTS + KV + SLOT CACHE.
# vLLM sizes the KV cache to fill --gpu-memory-utilization BEFORE Sluice
# allocates its slot buffers, so the slot cache is charged on top of a budget
# that is already spent. TP=2: 28.5 (weights) + ~40 (KV at 0.90) + 14.5 (slots)
# ~= 83 GiB against a 79 GiB card.
#
# This is exactly why the V4 runs needed gpu-util 0.55 for Sluice against 0.75
# for vanilla -- previously recorded as an environment quirk without the reason.
# There is no error message that explains it; the user just sees a CUDA OOM.
#
# 0.60 => budget 47.5 GiB => KV ~19 GiB, and 28.5 + 19 + 14.5 = 62 GiB, which
# fits with headroom. 19 GiB of KV is far more than 48 short sequences need, so
# the mismatch against the vanilla arms (0.90) does not bind on TPOT -- it only
# changes how much unused KV was reserved. Stated, not hidden.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/qtp2b.txt; : > "$RES"
M=Qwen/Qwen3-30B-A3B

serve () { local LOG="$1" UTIL="$2"; shift 2
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 2 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization "$UTIL" \
    --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE \
    "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 90); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 10
  done; return 1; }
bench () { local OUT
  OUT=$(timeout 1200 vllm bench serve --base-url http://localhost:8000 \
    --model $M --trust-remote-code --dataset-name random \
    --random-input-len 32 --random-output-len 200 --ignore-eos \
    --num-prompts "$3" --max-concurrency "$2" 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "$1 c=$2 | $OUT" >> "$RES"; }
arm () { local TAG="$1" LOG="$2" UTIL="$3"
  if serve "$LOG" "$UTIL"; then
    bench "warm" 6 24 >/dev/null; sed -i '/^warm c=6/d' "$RES"
    bench "$TAG" 1 8; bench "$TAG" 3 24; bench "$TAG" 6 48
    { grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$LOG" | head -1
      grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1
      grep -oE "GPU KV cache size: [0-9,]+" "$LOG" | head -1
    } | sed 's/^/      /' >> "$RES"
  else
    echo "$TAG | STARTUP FAILED" >> "$RES"
    grep -oE "(ValueError|torch.OutOfMemoryError): [^\"]{0,100}" "$LOG" | tail -1 | sed 's/^/      /' >> "$RES"
  fi; }

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=64
arm "TP2 SLUICE slots=64 (util .60)" /work/t2b_sluice.log 0.60

# matched vanilla at the SAME lowered util, so the pair is clean on every flag
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS
arm "TP2 vanilla_matched (util .60)" /work/t2b_match.log 0.60

pkill -f 'bin/vllm' 2>/dev/null
echo "QTP2B_DONE" >> "$RES"
