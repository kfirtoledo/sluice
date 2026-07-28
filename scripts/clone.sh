#!/bin/bash
# H-KERNEL [3]: is the 24.02 ms at 0 breaks the hidden_states.clone()?
# Arms are all measured on THIS instance so the N=0 clone control is
# within-instance (entries [1]/[2] ran on different nodes).
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
export SLUICE_SLOTS=48 SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1
export SLUICE_RS_FAST_HIT=1 SLUICE_RS_NOOP=1
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/clone.txt; : > "$RES"

# arm = "<BREAK_LAYERS> <NO_CLONE> <tag>"
run_arm () {
  local N="$1" NC="$2" TAG="$3"
  local LOG=/work/cl_$TAG.log
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  SLUICE_BREAK_LAYERS=$N SLUICE_NO_CLONE=$NC nohup setsid \
    vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.55 \
    --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE \
    >"$LOG" 2>&1 </dev/null &
  local ok=0 i
  for i in $(seq 1 75); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && { ok=1; break; }
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && break
    sleep 15
  done
  if [ $ok -ne 1 ]; then
    echo "$TAG (N=$N NC=$NC) | STARTUP FAILED" >> "$RES"
    tail -20 "$LOG" >> "$RES"
    return
  fi
  local r OUT
  for r in warm r1 r2; do
    OUT=$(timeout 900 vllm bench serve --base-url http://localhost:8000 \
      --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
      --dataset-name random --random-input-len 32 --random-output-len 200 \
      --ignore-eos --num-prompts 64 --max-concurrency 8 2>&1 \
      | grep -E 'Mean TPOT|Output token throughput' | tr '\n' ' ')
    echo "$TAG (N=$N NC=$NC) $r | $OUT" >> "$RES"
  done
}

run_arm 0  0 "n0_clone"     # control, reproduces the 33.85 ms reference
run_arm 0  1 "n0_noclone"   # DECISIVE: gap op becomes a pure passthrough
run_arm -1 1 "n43_noclone"  # cross-check at the normal break count
pkill -f 'bin/vllm' 2>/dev/null
echo "CLONE_DONE" >> "$RES"
