#!/bin/bash
# pds-perf [3c]: Qwen3-30B-A3B at TP=2 — the config where Sluice actually fits.
#
# WHY TP=2. At TP=1 Sluice OOMs on this model at slots=64 AND slots=48:
#   weights ~57 GiB + slot buffers (0.45 GiB per slot) exceeds the 79 GiB card,
#   because Sluice's peak is weights + slot cache. slots=48 left 318 MiB free.
#   Cutting slots far enough to fit drops the envelope (slots//top_k, top_k=8)
#   below 6, so c=6 would no longer take the split path -- the arm would stop
#   being comparable across the very concurrencies under test.
# TP=2 halves per-rank weights to ~28.5 GiB and slots=64 to ~14.5 GiB: peak ~43
# GiB, envelope back to 64//8 = 8, and c=1/3/6 all take the traced split path.
#
# ALL arms re-run at TP=2 so nothing is compared across topologies. The TP=1
# numbers already collected stay as a separate table.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/qtp2.txt; : > "$RES"
M=Qwen/Qwen3-30B-A3B

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 2 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization 0.90 \
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
arm () { local TAG="$1" LOG="$2"; shift 2
  if serve "$LOG" "$@"; then
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
arm "TP2 vanilla_stock  " /work/t2_stock.log
arm "TP2 vanilla_matched" /work/t2_match.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=64
arm "TP2 SLUICE slots=64" /work/t2_sluice.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS
arm "TP2 vanilla_stock(last)" /work/t2_stock2.log     # drift control
pkill -f 'bin/vllm' 2>/dev/null
echo "QTP2_DONE" >> "$RES"
