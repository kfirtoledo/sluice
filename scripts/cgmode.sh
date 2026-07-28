#!/bin/bash
# H-KERNEL [5c]: the OTHER unmatched flag.
#
# The ledger's router-split bundle contains THREE things absent from its
# vanilla arm, not two:
#     SLUICE_* env      (the offloader itself)
#     --max-num-batched-tokens 8      (the envelope; [5b] measures its curve)
#     -cc.cudagraph_mode=PIECEWISE    <-- THIS. untested.
#
# On V4 vLLM forces VLLM_USE_BREAKABLE_CUDAGRAPH and compilation mode NONE, so
# PIECEWISE vs the default is not the usual torch.compile-splitting question --
# it changes how the breakable capture is segmented. It could plausibly carry
# the whole ~20 ms on its own.
#
# NO SLUICE IN ANY ARM. Crossed with MAX_BT so the two candidates separate:
# if only the MAX_BT rows move, it is the envelope; if only the mode rows
# move, the envelope is innocent and [5b]'s curve is a red herring.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/cgmode.txt; : > "$RES"

bench_arm () {
  local TAG="$1"; shift
  local LOG=/work/cg_$TAG.log
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  timeout 3600 vllm bench latency \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.75 \
    --input-len 8 --output-len 200 --batch-size 8 \
    --num-iters-warmup 10 --num-iters 30 "$@" \
    >"$LOG" 2>&1 </dev/null
  local AVG MODE
  AVG=$(grep -iE 'Avg latency' "$LOG" | tail -1 | tr -s ' ')
  MODE=$(grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$LOG" | tail -1)
  if [ -z "$AVG" ]; then
    echo "$TAG | FAILED: $(grep -oE 'ValueError: [^\"]*' "$LOG" | tail -1 | cut -c1-110)" >> "$RES"
    return
  fi
  echo "$TAG | $AVG | $MODE" >> "$RES"
}

# mode default (omit the flag) x MAX_BT {8, 512}. The PIECEWISE row is already
# measured by [5b] at the same gpu-util, so the 2x2 completes across scripts.
bench_arm defmode_bt8   --max-num-batched-tokens 8
bench_arm defmode_bt512 --max-num-batched-tokens 512
echo "CGMODE_DONE" >> "$RES"
