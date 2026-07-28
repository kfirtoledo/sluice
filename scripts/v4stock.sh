#!/bin/bash
# Supplement to [10]: V4 vanilla_stock, which cannot run at gpu-util 0.55.
# Uncapped max_num_batched_tokens needs more headroom than the Sluice arms
# leave ("No available memory for the cache blocks"), so this arm runs at 0.75.
# The mismatch is documented, not hidden -- and it is demonstrably harmless:
# the same stock config measured 9.81 ms at 0.75 here and 9.83 ms at 0.55 in
# the ledger, so gpu-util does not move this number.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
RES=/work/c1c8.txt
V4=deepseek-ai/DeepSeek-V4-Flash
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
LOG=/work/c_v4_stock75.log
pkill -f 'bin/vllm' 2>/dev/null; sleep 10
nohup setsid vllm serve $V4 --revision $REV --trust-remote-code --port 8000 \
  --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
  --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.75 \
  >"$LOG" 2>&1 </dev/null &
ok=0
for i in $(seq 1 75); do
  grep -q 'Application startup complete' "$LOG" 2>/dev/null && { ok=1; break; }
  grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && break
  sleep 15
done
if [ $ok -ne 1 ]; then echo "V4 vanilla_stock(0.75) | FAILED AGAIN" >> "$RES"; exit 1; fi
bench () {
  local OUT
  OUT=$(timeout 1200 vllm bench serve --base-url http://localhost:8000 \
    --model $V4 --trust-remote-code --dataset-name random \
    --random-input-len 32 --random-output-len 200 --ignore-eos \
    --num-prompts "$2" --max-concurrency "$1" 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "V4 vanilla_stock  c=$1 | $OUT   [gpu-util 0.75]" >> "$RES"
}
bench 8 16 >/dev/null 2>&1   # warm, discarded
bench 1 8
bench 8 64
grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$LOG" | head -1 >> "$RES"
pkill -f 'bin/vllm' 2>/dev/null
echo "V4STOCK_DONE" >> "$RES"
