#!/bin/bash
# pds-perf [3b]: the Qwen3 Sluice arm, retried at slots=48.
#
# slots=64 OOMed at TP=1: Qwen3-30B-A3B is ~57 GiB of weights and 64/128 experts
# is ~27 GiB of slot buffers, so the load peak (~84 GiB) exceeds the 79 GiB card.
# It was ~800 MiB short. slots=48 frees ~6.75 GiB of that.
#
# top-8 makes the envelope tight: slots//top_k = 48/8 = 6, so MAX_BT=6 covers
# c=1/3/6 exactly and every step still takes the traced split path.
#
# vanilla_matched is RE-RUN at MAX_BT=6 so the pair is matched -- the earlier
# matched arm used MAX_BT=8 and is not a valid control for this one.
# 40 % of experts offloaded to host RAM (48 of 128).
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/qwen48.txt; : > "$RES"
M=Qwen/Qwen3-30B-A3B

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 1 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization 0.90 \
    --max-num-batched-tokens 6 -cc.cudagraph_mode=PIECEWISE \
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
    bench "warm" 6 24 >/dev/null; sed -i '/^warm c=6/d' "$RES"
    bench "$TAG" 1 8; bench "$TAG" 3 24; bench "$TAG" 6 48
    { grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$LOG" | head -1
      grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1
      grep -oE "GPU KV cache size: [0-9,]+" "$LOG" | head -1
    } | sed 's/^/      /' >> "$RES"
  else
    echo "$TAG | STARTUP FAILED" >> "$RES"
    grep -oE "(ValueError|torch.OutOfMemoryError): [^\"]{0,100}" "$LOG" | tail -1 | sed 's/^/      /' >> "$RES"
  fi; }

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
arm "vanilla_matched(BT6)" /work/q48_match.log

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=48
arm "SLUICE slots=48     " /work/q48_sluice.log

pkill -f 'bin/vllm' 2>/dev/null
echo "QWEN48_DONE" >> "$RES"
