#!/bin/bash
# H-KERNEL [12]: does entry [8]'s FULL-cudagraph win transfer to V2-Lite?
#
# PREDICTION, recorded before the run: NO.
# V2-Lite is on the FX-SPLITTING path -- vLLM compiles it, and Sluice's gap
# exists BECAUSE vllm::moe_forward is added to splitting_ops. There, a FULL
# graph really would swallow the hook's D2H sync, so the guard's refusal is
# correct and SLUICE_ALLOW_FULL_CG should NOT engage (it requires compilation
# mode NONE and empty splitting_ops).
#
# Expected: arm B refuses to start with Sluice's "requires PIECEWISE cudagraph"
# RuntimeError, or falls back to PIECEWISE. If instead it starts AND runs
# faster, my model of the two paths is wrong and entry [8]'s scope is wider
# than claimed -- which is worth knowing either way.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/v2full.txt; : > "$RES"
M=deepseek-ai/DeepSeek-V2-Lite

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 1 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization 0.50 \
    --max-num-batched-tokens 8 "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 60); do
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
info () {
  grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$1" | head -1 >> "$RES"
  grep -oE "'mode': <CompilationMode\.[A-Z_]+" "$1" | head -1 >> "$RES"
  grep -oE "'splitting_ops': \[[^]]{0,60}" "$1" | head -1 >> "$RES"
  grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$1" | head -1 >> "$RES"; }

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=48

# arm A: control -- today's shipping V2-Lite config (forced PIECEWISE)
export SLUICE_ALLOW_FULL_CG=0
if serve /work/v2f_pw.log -cc.cudagraph_mode=PIECEWISE; then
  bench "warm" 8 16 >/dev/null; sed -i '/^warm /d' "$RES"
  bench "V2 SLUICE PIECEWISE(control)" 1 8; bench "V2 SLUICE PIECEWISE(control)" 8 64; info /work/v2f_pw.log
else echo "V2 SLUICE PIECEWISE | STARTUP FAILED" >> "$RES"; fi

# arm B: drop the flag and ask for FULL. Expected to be REFUSED on this path.
export SLUICE_ALLOW_FULL_CG=1
if serve /work/v2f_full.log; then
  bench "warm" 8 16 >/dev/null; sed -i '/^warm /d' "$RES"
  bench "V2 SLUICE default-cg   " 1 8; bench "V2 SLUICE default-cg   " 8 64; info /work/v2f_full.log
else
  echo "V2 SLUICE default-cg | REFUSED/FAILED (expected on the fx path):" >> "$RES"
  grep -oE "Sluice: [^\"]{0,190}" /work/v2f_full.log | tail -1 >> "$RES"
fi
unset SLUICE_ALLOW_FULL_CG; pkill -f 'bin/vllm' 2>/dev/null
echo "V2FULL_DONE" >> "$RES"
