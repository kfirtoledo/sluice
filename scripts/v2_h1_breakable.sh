#!/bin/bash
# PDS-PERF H1: transfer the FULL-cudagraph win to V2-Lite via breakable capture.
#
# V4 gained -28% TPOT by keeping vLLM's FULL decode graphs. V2-Lite cannot
# today: it is on the FX-SPLITTING path (compilation mode VLLM_COMPILE,
# splitting_ops = attention ops), where Sluice's guard correctly refuses FULL --
# there the eager gap exists ONLY because moe_forward is in splitting_ops.
#
# But VLLM_USE_BREAKABLE_CUDAGRAPH is USER-SETTABLE FOR ANY MODEL. vLLM only
# AUTO-enables it for V4/MiniMax architectures (vllm/config/vllm.py:1113-1127).
# Setting it on V2-Lite gives compilation mode NONE, empty splitting_ops, and a
# gap from add_eager -- exactly the conditions SLUICE_ALLOW_FULL_CG=1 requires.
#
# PRE-REGISTERED PREDICTION (written before running):
#   Breakable disables torch.compile ENTIRELY. On V4 that was already true, so
#   FULL was pure gain. On V2-Lite, Inductor may be worth more than FULL.
#   Honest prior: ~50/50. V2-Lite is small (26 layers, TP=1), so per-step launch
#   overhead -- which is what FULL removes -- matters less than it does on V4's
#   43 layers x 4 ranks. Arm C exists to tell "breakable helps" apart from
#   "Sluice benefits from breakable".
#   DECISION RULE: arm B >= 5% better than arm A => pursue. Worse => close.
#
# Baseline to beat (measured 2026-07-27, same workload): SLUICE 13.74 ms / 483 tok/s.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/h1.txt; : > "$RES"
M=deepseek-ai/DeepSeek-V2-Lite

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 1 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization 0.50 \
    --max-num-batched-tokens 8 "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 70); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 10
  done; return 1; }
bench () { local OUT
  OUT=$(timeout 900 vllm bench serve --base-url http://localhost:8000 \
    --model $M --trust-remote-code --dataset-name random \
    --random-input-len 32 --random-output-len 200 --ignore-eos \
    --num-prompts 64 --max-concurrency 8 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "$1 | $OUT" >> "$RES"; }
info () { { grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$1" | head -1
            grep -oE "'mode': <CompilationMode\.[A-Z_]+" "$1" | head -1
            grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$1" | head -1
            grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$1" | tail -1
          } | sed 's/^/      /' >> "$RES"; }
arm () { local TAG="$1" LOG="$2"; shift 2
  if serve "$LOG" "$@"; then
    bench "warm(discard)"; sed -i '/^warm(discard)/d' "$RES"
    bench "$TAG r1"; bench "$TAG r2"; bench "$TAG r3"; info "$LOG"
  else
    echo "$TAG | FAILED" >> "$RES"
    grep -oE "Sluice: [^\"]{0,150}|(ValueError|RuntimeError|torch.OutOfMemoryError): [^\"]{0,110}" "$LOG" | tail -2 | sed 's/^/      /' >> "$RES"
  fi; }

SL="SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=48"

# A: today's shipping config (fx path, forced PIECEWISE)
export $SL SLUICE_ALLOW_FULL_CG=0; unset VLLM_USE_BREAKABLE_CUDAGRAPH
arm "A sluice fx+PIECEWISE " /work/h1_a.log -cc.cudagraph_mode=PIECEWISE

# B: breakable path + FULL decode graphs  <-- the hypothesis
export $SL SLUICE_ALLOW_FULL_CG=1 VLLM_USE_BREAKABLE_CUDAGRAPH=1
arm "B sluice breakable+FULL" /work/h1_b.log

# C: vanilla under breakable -- separates "breakable helps" from "Sluice gains"
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
arm "C vanilla breakable    " /work/h1_c.log

# D: vanilla on the fx path -- the matched control for A
unset VLLM_USE_BREAKABLE_CUDAGRAPH
arm "D vanilla fx+PIECEWISE " /work/h1_d.log -cc.cudagraph_mode=PIECEWISE

pkill -f 'bin/vllm' 2>/dev/null
echo "H1_DONE" >> "$RES"
