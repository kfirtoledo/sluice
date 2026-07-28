#!/bin/bash
# [10-V2]: DeepSeek-V2-Lite, Sluice vs vanilla, c=1 and c=8. Own pod (1 GPU).
#
# V2-Lite is on the FX-SPLITTING path: the eager gap exists BECAUSE
# vllm::moe_forward is in splitting_ops, so forcing PIECEWISE is genuinely
# required here and entry [8]'s FULL win should NOT transfer. SLUICE_ALLOW_FULL_CG
# is set anyway -- if the guard engages, the arm's logged cudagraph_mode will
# say so and my model of the two paths is wrong.
#
# Two baselines, as everywhere in this campaign:
#   vanilla_stock   - vLLM defaults; the real "should I use Sluice" comparison
#   vanilla_matched - vanilla carrying Sluice's flags; isolates Sluice's own cost
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/v2.txt; : > "$RES"
M=deepseek-ai/DeepSeek-V2-Lite

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 1 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization 0.50 \
    "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 60); do
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

arm () {  # $1 tag  $2 log  ; rest = flags
  local TAG="$1" LOG="$2"; shift 2
  if serve "$LOG" "$@"; then
    bench "warm" 8 16 > /dev/null; sed -i '/^warm c=8/d' "$RES"
    bench "$TAG" 1 8
    bench "$TAG" 8 64
    grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$LOG" | head -1 >> "$RES"
    grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$LOG" | head -1 >> "$RES"
    grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1 >> "$RES"
  else
    echo "$TAG | STARTUP FAILED: $(grep -oE '(ValueError|RuntimeError|torch.OutOfMemoryError): [^\"]{0,100}' "$LOG" | tail -1)" >> "$RES"
  fi; }

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
arm "V2 vanilla_stock  " /work/v2_stock.log
arm "V2 vanilla_matched" /work/v2_match.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE
export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=48 SLUICE_ALLOW_FULL_CG=1
arm "V2 SLUICE slots=48" /work/v2_sluice.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE
pkill -f 'bin/vllm' 2>/dev/null
echo "V2_DONE" >> "$RES"
