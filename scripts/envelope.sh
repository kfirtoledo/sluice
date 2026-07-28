#!/bin/bash
# H-KERNEL [5b]: the ENVELOPE CURVE. What does capping the batch at
# max_num_batched_tokens cost per decode step?
#
# This is now the campaign's central number. Router-split requires
# MAX_BT <= slots//top_k (48/6 = 8 on V4), and entry [4] showed that cap --
# not the graph restructuring -- is what separates the Sluice arms from the
# ledger's vanilla reference.
#
# NO SLUICE IN ANY ARM. Pure vanilla; MAX_BT is the only variable, so the
# curve is a property of vLLM's scheduler/capture, not of the offloader.
#
# gpu-memory-utilization raised 0.55 -> 0.75 for ALL arms: at 0.55 the
# uncapped arm dies with "No available memory for the cache blocks" (a
# profiling-headroom failure, not the effect under test). Held constant
# across arms so it cannot confound the comparison. Absolute numbers are
# therefore NOT comparable to entries [3]/[4]; the shape of the curve is.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/envelope.txt; : > "$RES"

bench_arm () {
  local BT="$1"
  local LOG=/work/env_bt$BT.log
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  timeout 3600 vllm bench latency \
    --model deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.75 \
    -cc.cudagraph_mode=PIECEWISE --max-num-batched-tokens "$BT" \
    --input-len 8 --output-len 200 --batch-size 8 \
    --num-iters-warmup 10 --num-iters 30 \
    >"$LOG" 2>&1 </dev/null
  local AVG CG
  AVG=$(grep -iE 'Avg latency' "$LOG" | tail -1 | tr -s ' ')
  CG=$(grep -oE "'cudagraph_capture_sizes': \[[^]]*\]" "$LOG" | tail -1)
  if [ -z "$AVG" ]; then
    echo "MAX_BT=$BT | FAILED: $(grep -oE 'ValueError: [^\"]*' "$LOG" | tail -1)" >> "$RES"
    return
  fi
  echo "MAX_BT=$BT | $AVG | $CG" >> "$RES"
}

for BT in 8 16 32 128 512 2048; do bench_arm "$BT"; done
echo "ENVELOPE_DONE" >> "$RES"
