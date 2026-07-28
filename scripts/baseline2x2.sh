#!/bin/bash
# H-KERNEL [5]: WHICH flag makes vanilla look 3x faster in the ledger?
#
# The ledger's vanilla@4 reference (9.83 ms TPOT / 786 tok/s) ran with vLLM
# DEFAULTS. Every Sluice arm -- including every no-op floor the "graph-shape"
# term was derived from -- ran with two flags the router-split path requires:
#
#   --moe-backend marlin        (Sluice re-points slot buffers; marlin layout)
#   --max-num-batched-tokens 8  (router-split's smallness envelope, slots//topk)
#
# vanilla WITH both measures ~28.9 ms here, i.e. essentially the whole gap.
# This 2x2 attributes it. NO SLUICE IN ANY ARM -- pure vanilla throughout, so
# the only variables are the two flags.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/baseline2x2.txt; : > "$RES"

bench_arm () {
  local TAG="$1"; shift            # remaining args: extra serve flags
  local LOG=/work/b2_$TAG.log
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  timeout 3600 vllm bench latency \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code \
    --tensor-parallel-size 4 --enable-expert-parallel \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.55 \
    -cc.cudagraph_mode=PIECEWISE \
    --input-len 8 --output-len 200 --batch-size 8 \
    --num-iters-warmup 10 --num-iters 30 "$@" \
    >"$LOG" 2>&1 </dev/null
  local AVG BK
  AVG=$(grep -iE 'Avg latency' "$LOG" | tail -1 | tr -s ' ')
  BK=$(grep -oE "moe_backend='[a-z_]+'" "$LOG" | tail -1)
  if [ -z "$AVG" ]; then
    echo "$TAG | FAILED" >> "$RES"; tail -12 "$LOG" >> "$RES"; return
  fi
  echo "$TAG | $AVG | $BK" >> "$RES"
}

# 2x2: {marlin, default backend} x {MAX_BT=8, MAX_BT default}
bench_arm marlin_bt8    --moe-backend marlin --max-num-batched-tokens 8
bench_arm marlin_btdef  --moe-backend marlin
bench_arm defbe_bt8     --max-num-batched-tokens 8
bench_arm defbe_btdef
echo "BASELINE2X2_DONE" >> "$RES"
