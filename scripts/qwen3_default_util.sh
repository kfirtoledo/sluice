#!/bin/bash
# pds-perf [4]: Qwen3-30B-A3B, TP=2, **vLLM DEFAULT gpu_memory_utilization**.
#
# No --gpu-memory-utilization anywhere. Earlier arms hand-tuned it (0.60 for
# Sluice, 0.90 for vanilla) which invites the objection that the comparison was
# shaped by memory settings. Every arm now uses the same default.
#
# Sluice OOMed previously at 0.90. My allocator-ordering explanation does NOT
# hold up: post_init already calls torch.accelerator.empty_cache() per layer
# (offloader.py:636), so freed expert weights ARE returned to the driver. Since
# I cannot explain it from the source, this run captures the FULL traceback so
# the failing phase is identifiable rather than guessed at.
#
# slots=96 added (25 % offload, envelope 96//8 = 12) alongside slots=64
# (50 % offload, envelope 8).
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/qdef.txt; : > "$RES"
M=Qwen/Qwen3-30B-A3B

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 2 --moe-backend triton \
    --max-model-len 2048 "$@" >"$LOG" 2>&1 </dev/null &
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
      grep -oiE "Sluice[^\"]{0,90}(VRAM|host|GiB)[^\"]{0,40}" "$LOG" | head -2
    } | sed 's/^/      /' >> "$RES"
  else
    echo "$TAG | STARTUP FAILED — failing phase below" >> "$RES"
    # which lifecycle phase did it die in? this is the diagnostic that matters
    grep -nE "Memory profiling|determine_available_memory|Capturing CUDA graphs|init_device|load_model|post_init|_install_cache" "$LOG" | tail -4 | sed 's/^/      ctx: /' >> "$RES"
    grep -E "torch.OutOfMemoryError|ValueError: No available memory" "$LOG" | tail -1 | cut -c1-150 | sed 's/^/      err: /' >> "$RES"
    grep -B2 -A12 "OutOfMemoryError" "$LOG" | grep -oE "File \"[^\"]+\", line [0-9]+, in [a-z_]+" | tail -6 | sed 's/^/      at: /' >> "$RES"
  fi; }

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
arm "DEF vanilla_stock  " /work/d_stock.log
arm "DEF vanilla_matched" /work/d_match.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1
export SLUICE_SLOTS=64
arm "DEF SLUICE slots=64" /work/d_sl64.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE
export SLUICE_SLOTS=96
arm "DEF SLUICE slots=96" /work/d_sl96.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS
arm "DEF vanilla_stock(last)" /work/d_stock2.log
pkill -f 'bin/vllm' 2>/dev/null
echo "QDEF_DONE" >> "$RES"
