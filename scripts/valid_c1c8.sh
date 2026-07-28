#!/bin/bash
# DSV2-Lite + Qwen3-30B-A3B at c=1 and c=8, on the VALID configuration.
#
# PR#5 / issue #4: router-split emits INVALID OUTPUT when Inductor compilation
# and CUDA-graph capture are both active. VLLM_USE_BREAKABLE_CUDAGRAPH=1 is the
# fix -- it forces compilation mode NONE while keeping capture.
#
# Every V2-Lite number I measured yesterday used mode=VLLM_COMPILE +
# cudagraph_mode=PIECEWISE, i.e. exactly the invalid combination. This re-runs
# them on the valid one. Numbers are expected to be WORSE; that is the point.
#
# On breakable, cudagraph_mode defaults to FULL_AND_PIECEWISE, which Sluice
# refuses unless permitted. TP=1 here, so BOTH mechanisms are available:
# SLUICE_OPTIMISTIC_FULL (single-rank only) and SLUICE_ALLOW_FULL_CG
# (any TP, breakable only). Arms for both, since this is the one configuration
# where they can be compared directly.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/valid.txt; : > "$RES"

serve () { local LOG="$1" MODEL="$2" TP="$3"; shift 3
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  nohup setsid vllm serve "$MODEL" --trust-remote-code --port 8000 \
    --tensor-parallel-size "$TP" --moe-backend triton \
    --max-model-len 2048 --max-num-batched-tokens 8 \
    "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 80); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 10
  done; return 1; }
bench () { local OUT
  OUT=$(timeout 1200 vllm bench serve --base-url http://localhost:8000 \
    --model "$2" --trust-remote-code --dataset-name random \
    --random-input-len 32 --random-output-len 200 --ignore-eos \
    --num-prompts "$4" --max-concurrency "$3" 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "$1 c=$3 | $OUT" >> "$RES"; }
arm () { local TAG="$1" LOG="$2" MODEL="$3" TP="$4"; shift 4
  if serve "$LOG" "$MODEL" "$TP" "$@"; then
    bench "warm" "$MODEL" 8 16 >/dev/null; sed -i '/^warm c=8/d' "$RES"
    bench "$TAG" "$MODEL" 1 8
    bench "$TAG" "$MODEL" 8 64
    { grep -oE "'mode': <CompilationMode\.[A-Z_]+" "$LOG" | head -1
      grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$LOG" | head -1
      grep -oE "permitting cudagraph_mode=[A-Z_]+ via [A-Za-z_()]+" "$LOG" | head -1
      grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$LOG" | head -1
      grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1
    } | sed 's/^/      /' >> "$RES"
  else
    echo "$TAG | STARTUP FAILED" >> "$RES"
    grep -oE "Sluice: [^\"]{0,140}|(ValueError|RuntimeError|torch.OutOfMemoryError): [^\"]{0,110}" "$LOG" | tail -2 | sed 's/^/      /' >> "$RES"
  fi; }

SL="SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=48"
V2=deepseek-ai/DeepSeek-V2-Lite

echo "=== DeepSeek-V2-Lite, TP=1, VALID config (breakable) ===" >> "$RES"
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG SLUICE_OPTIMISTIC_FULL
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
arm "V2 vanilla        " /work/v_v2_van.log  $V2 1 --gpu-memory-utilization 0.50
export $SL SLUICE_ALLOW_FULL_CG=1
arm "V2 SLUICE allow-cg" /work/v_v2_acg.log  $V2 1 --gpu-memory-utilization 0.50
unset SLUICE_ALLOW_FULL_CG; export $SL SLUICE_OPTIMISTIC_FULL=1
arm "V2 SLUICE optimist" /work/v_v2_opt.log  $V2 1 --gpu-memory-utilization 0.50
echo "VALID_V2_DONE" >> "$RES"
