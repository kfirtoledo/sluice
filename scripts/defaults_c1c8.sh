#!/bin/bash
# V2-Lite + Qwen3-30B-A3B at c=1 and c=8, vLLM DEFAULTS wherever possible.
#
# NO --gpu-memory-utilization anywhere: the memory-accounting fix in this
# branch makes vLLM see the slot cache, so Sluice no longer needs a hand-tuned
# value. That flag was a workaround for a bug, and the bug is fixed.
#
# --max-num-batched-tokens CANNOT be dropped for Sluice. router-split is
# single-wave by contract, so MBT <= slots//top_k, and relaxing it under
# capture was tried on 2026-07-25: the config starts and then dies mid-run
# ("streaming hook reached under CUDA graph capture") because the
# oversized-step fallback IS the classic hook, which refuses to run captured.
# It is set ONLY on Sluice arms, and vanilla runs without it -- so the
# comparison charges Sluice for the flag it forces, which is the honest way
# round. Both baselines are reported.
#
# VLLM_USE_BREAKABLE_CUDAGRAPH=1 on every arm of these models: without it
# router-split emits invalid output (issue #4). Vanilla gets it too so the
# capture path is matched.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
RES=/work/def.txt; : > "$RES"

serve () { local LOG="$1" MODEL="$2" TP="$3"; shift 3
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve "$MODEL" --trust-remote-code --port 8000 \
    --tensor-parallel-size "$TP" --max-model-len 2048 "$@" \
    >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 90); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 10
  done; return 1; }
bench () { local OUT
  OUT=$(timeout 1500 vllm bench serve --base-url http://localhost:8000 \
    --model "$2" --trust-remote-code --dataset-name random \
    --random-input-len 32 --random-output-len 200 --ignore-eos \
    --num-prompts "$4" --max-concurrency "$3" 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "$1 c=$3 | $OUT" >> "$RES"; }
arm () { local TAG="$1" LOG="$2" MODEL="$3" TP="$4"; shift 4
  if serve "$LOG" "$MODEL" "$TP" "$@"; then
    bench "w" "$MODEL" 8 16 >/dev/null; sed -i '/^w c=8/d' "$RES"
    bench "$TAG" "$MODEL" 1 8
    bench "$TAG" "$MODEL" 8 64
    { grep -oE "'mode': <CompilationMode\.[A-Z_]+|'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$LOG" | head -2
      grep -oE "permitting cudagraph_mode=[A-Z_]+ via [A-Za-z_()]+" "$LOG" | head -1
      grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$LOG" | head -1
      grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1
      grep -oE "Available KV cache memory: [0-9.]+ GiB" "$LOG" | head -1
    } | sed 's/^/      /' >> "$RES"
  else
    echo "$TAG | STARTUP FAILED" >> "$RES"
    grep -oE "Sluice: [^\"]{0,140}|(ValueError|RuntimeError|torch.OutOfMemoryError): [^\"]{0,110}" "$LOG" | tail -2 | sed 's/^/      /' >> "$RES"
  fi; }

SL="SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_ALLOW_FULL_CG=1"
V2=deepseek-ai/DeepSeek-V2-Lite
Q3=Qwen/Qwen3-30B-A3B

echo "=== DeepSeek-V2-Lite TP=1 (vLLM defaults; Sluice adds only MBT=8) ===" >> "$RES"
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
arm "V2 vanilla_default" /work/d_v2_van.log $V2 1
export $SL SLUICE_SLOTS=48
arm "V2 SLUICE slots=48" /work/d_v2_sl.log  $V2 1 --max-num-batched-tokens 8

echo "=== Qwen3-30B-A3B TP=2 (vLLM defaults; Sluice adds only MBT) ===" >> "$RES"
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
arm "Q3 vanilla_default" /work/d_q3_van.log $Q3 2
export $SL SLUICE_SLOTS=64
arm "Q3 SLUICE slots=64" /work/d_q3_sl.log  $Q3 2 --max-num-batched-tokens 8
echo "DEFAULTS_DONE" >> "$RES"
