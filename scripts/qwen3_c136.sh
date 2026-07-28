#!/bin/bash
# pds-perf [3]: Qwen3-30B-A3B at c=1, 3, 6 — a THIRD model for the comparison.
#
# Why this model is worth having: it is a different point in the design space
# from both DeepSeek models measured so far.
#   Qwen3-30B-A3B : 48 layers, 128 experts, top-8, bf16, fits on ONE H100
#   V2-Lite       : 26 layers,  64 experts, top-6, bf16, TP=1
#   V4-Flash      : 43 layers, 256 experts, top-6, fp8,  TP=4 + EP
# Sluice's envelope is slots//top_k, and top-8 makes it TIGHTER per slot than
# either DeepSeek model: slots=64 buys an envelope of only 8 tokens.
#
# Qwen3 is on the FX-SPLITTING path (like V2-Lite), so forced PIECEWISE and the
# entry [8] FULL win does NOT apply. Recorded, not assumed — each arm logs the
# compilation mode it actually got.
#
# Two baselines per concurrency, as everywhere in this campaign:
#   vanilla_stock   = vLLM defaults ("should I use Sluice?")
#   vanilla_matched = Sluice's flags, no Sluice ("what does Sluice itself cost?")
#
# slots=64 of 128 experts = 50 % offloaded to host RAM.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/qwen.txt; : > "$RES"
M=Qwen/Qwen3-30B-A3B

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 1 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization 0.90 \
    "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 90); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 10
  done; return 1; }

bench () {  # $1 tag  $2 concurrency  $3 prompts
  local OUT
  OUT=$(timeout 1200 vllm bench serve --base-url http://localhost:8000 \
    --model $M --trust-remote-code --dataset-name random \
    --random-input-len 32 --random-output-len 200 --ignore-eos \
    --num-prompts "$3" --max-concurrency "$2" 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "$1 c=$2 | $OUT" >> "$RES"; }

arm () {  # $1 tag  $2 log ; rest = serve flags
  local TAG="$1" LOG="$2"; shift 2
  if serve "$LOG" "$@"; then
    bench "warm" 6 24 >/dev/null; sed -i '/^warm c=6/d' "$RES"
    bench "$TAG" 1 8
    bench "$TAG" 3 24
    bench "$TAG" 6 48
    { grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$LOG" | head -1
      grep -oE "'mode': <CompilationMode\.[A-Z_]+" "$LOG" | head -1
      grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$LOG" | head -1
      grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1
      grep -oE "GPU KV cache size: [0-9,]+" "$LOG" | head -1
    } | sed 's/^/      /' >> "$RES"
  else
    echo "$TAG | STARTUP FAILED" >> "$RES"
    grep -oE "Sluice: [^\"]{0,150}|(ValueError|RuntimeError|torch.OutOfMemoryError): [^\"]{0,110}" "$LOG" | tail -2 | sed 's/^/      /' >> "$RES"
  fi; }

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
arm "vanilla_stock  " /work/q_stock.log
arm "vanilla_matched" /work/q_match.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=64 SLUICE_ALLOW_FULL_CG=1
arm "SLUICE slots=64" /work/q_sluice.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
arm "vanilla_stock(last)" /work/q_stock2.log      # drift control
pkill -f 'bin/vllm' 2>/dev/null
echo "QWEN_DONE" >> "$RES"
