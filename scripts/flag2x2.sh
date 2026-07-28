#!/bin/bash
# H-KERNEL [7]: WHICH of the two envelope flags carries the ~19.3 ms?
#
# Entry [6] proved the pair costs +19.28 ms with no Sluice in the process, but
# the two have only ever moved together. This is the full 2x2:
#
#                    | no PIECEWISE      | PIECEWISE
#   MAX_BT default   | A (=9.83 known)   | E
#   MAX_BT 8         | D                 | B (=29.1 known)
#
# All four arms in ONE script at ONE gpu-util, so nothing is compared across
# scripts or instruments. A and B are re-measured here rather than reused.
# NO SLUICE IN ANY ARM.
#
# gpu-util 0.75: the uncapped arms (A, E) OOM at 0.55. Held constant.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/flag2x2.txt; : > "$RES"

run_arm () {
  local TAG="$1"; shift
  local LOG=/work/f2_$TAG.log
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.75 \
    "$@" >"$LOG" 2>&1 </dev/null &
  local ok=0 i
  for i in $(seq 1 75); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && { ok=1; break; }
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && break
    sleep 15
  done
  if [ $ok -ne 1 ]; then
    echo "$TAG | STARTUP FAILED: $(grep -oE '(ValueError|torch.OutOfMemoryError): [^\"]*' "$LOG" | tail -1 | cut -c1-90)" >> "$RES"
    return
  fi
  local r OUT
  for r in warm r1 r2 r3; do
    OUT=$(timeout 900 vllm bench serve --base-url http://localhost:8000 \
      --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
      --dataset-name random --random-input-len 32 --random-output-len 200 \
      --ignore-eos --num-prompts 64 --max-concurrency 8 2>&1 \
      | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
    echo "$TAG $r | $OUT" >> "$RES"
  done
}

run_arm A_neither
run_arm D_maxbt_only   --max-num-batched-tokens 8
run_arm E_piecewise_only                          -cc.cudagraph_mode=PIECEWISE
run_arm B_both         --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE
pkill -f 'bin/vllm' 2>/dev/null
echo "FLAG2X2_DONE" >> "$RES"
