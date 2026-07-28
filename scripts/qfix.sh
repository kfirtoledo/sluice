#!/bin/bash
# pds-perf [5]: does reporting the slot cache to vLLM's profiler fix the OOM?
#
# FIX UNDER TEST: the plugin's load_model wrapper now adds the slot-cache bytes
# to runner.model_memory_usage, after post_init and before
# determine_available_memory.
#
# PRE-REGISTERED PREDICTION, written before the run:
#   Before the fix, vLLM reported "Available KV cache memory: 70.29 GiB"
#   (slots=64) and 70.58 GiB (slots=96) and OOMed. If the diagnosis is right,
#   the reported figure must now DROP BY ROUGHLY THE SLOT SIZE and the two slot
#   counts must SEPARATE:
#       slots=64 -> ~13.5 GiB of slots  => available ~= 56-57 GiB
#       slots=96 -> ~20.3 GiB of slots  => available ~= 50 GiB
#   and both arms must START. If they start but the figures do NOT separate,
#   the fix is wrong even if the OOM disappears -- so the log line is checked,
#   not just the exit status.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/qfix.txt; : > "$RES"
M=Qwen/Qwen3-30B-A3B

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 2 --moe-backend triton --max-model-len 2048 \
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
arm () { local TAG="$1" LOG="$2"
  if serve "$LOG"; then
    echo "$TAG | STARTED" >> "$RES"
    bench "warm" 6 24 >/dev/null; sed -i '/^warm c=6/d' "$RES"
    bench "$TAG" 1 8; bench "$TAG" 3 24; bench "$TAG" 6 48
  else
    echo "$TAG | STILL FAILS" >> "$RES"
    grep -E "torch.OutOfMemoryError|No available memory" "$LOG" | tail -1 | cut -c1-120 | sed 's/^/      /' >> "$RES"
  fi
  { grep -oE "Available KV cache memory: [0-9.]+ GiB" "$LOG" | head -1
    grep -oE "reported [0-9.]+ GiB of GPU slot cache[^\"]{0,70}" "$LOG" | head -1
    grep -oE "GPU KV cache size: [0-9,]+" "$LOG" | head -1
    grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1
  } | sed 's/^/      /' >> "$RES"; }

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1
export SLUICE_SLOTS=64; arm "FIX SLUICE slots=64" /work/f_sl64.log
export SLUICE_SLOTS=96; arm "FIX SLUICE slots=96" /work/f_sl96.log
pkill -f 'bin/vllm' 2>/dev/null
echo "QFIX_DONE" >> "$RES"
