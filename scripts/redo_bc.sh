#!/bin/bash
# H-KERNEL [6b]: complete the triangle at the LEDGER's gpu-memory-utilization.
# Arm C OOMs at 0.75 (Sluice's slot buffers leave no room for KV), so B and C
# are both re-run at 0.55 -- the value the ledger used -- making them a fully
# matched pair. Arm A already reproduced the ledger's 9.83 ms at 0.75, which
# shows gpu-util is not what moves these numbers.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/redo_bc.txt; : > "$RES"

run_arm () {
  local TAG="$1"; shift
  local LOG=/work/rb_$TAG.log
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  env "$@" nohup setsid vllm serve deepseek-ai/DeepSeek-V4-Flash \
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
    echo "$TAG | STARTUP FAILED" >> "$RES"; return
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
  echo "$TAG | ARMED: $(grep -oE 'router-split layers=[0-9]+' "$LOG" | sort -u | tr '\n' ' ')" >> "$RES"
}

SL="SLUICE_SLOTS=48 SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_RS_NOOP=1"
run_arm B55_vanilla_matched IGNORE=1
run_arm C55_routersplit_noop $SL
pkill -f 'bin/vllm' 2>/dev/null
echo "REDO_BC_DONE" >> "$RES"
