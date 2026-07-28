#!/bin/bash
# H-KERNEL [13]: EP OFF -- the only way to widen the envelope, now combined
# with entry [8]'s FULL decode graphs.
#
# With EP on at TP=4, each rank holds 256/4 = 64 experts per layer, so
# slots <= 64 and the envelope caps at 64//6 = 10 tokens. With EP OFF each rank
# holds a TP-sharded copy of ALL 256 experts, so slots can reach ~192 and the
# envelope opens to 192//6 = 32 tokens.
#
# The ledger measured one EP-off point (slots=192, env 32) and called it "not a
# free win: at c=8 it is worse (54.81 vs 44.83)". But that judgement predates
# BOTH corrections: it was scored against an unmatched vanilla, and it ran with
# forced PIECEWISE. Re-measured here with a matched baseline and FULL graphs.
#
# c=8 AND c=32: the envelope only pays off when the batch actually exceeds 8
# tokens, which is precisely what c=8 cannot show.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/epoff.txt; : > "$RES"

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 4 --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 90); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 15
  done; return 1; }
bench () { local OUT
  OUT=$(timeout 1800 vllm bench serve --base-url http://localhost:8000 \
    --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code --dataset-name random \
    --random-input-len 32 --random-output-len 200 --ignore-eos \
    --num-prompts "$3" --max-concurrency "$2" 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "$1 c=$2 | $OUT" >> "$RES"; }
info () { grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$1" | head -1 >> "$RES"
          grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$1" | head -1 >> "$RES"
          grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$1" | tail -1 >> "$RES"; }
run () { local TAG="$1" LOG="$2"; shift 2
  if serve "$LOG" "$@"; then
    bench "warm" 8 16 >/dev/null; sed -i '/^warm /d' "$RES"
    bench "$TAG" 8 64; bench "$TAG" 32 128; info "$LOG"
  else echo "$TAG | FAILED: $(grep -oE '(ValueError|torch.OutOfMemoryError): [^\"]{0,90}' "$LOG" | tail -1)" >> "$RES"; fi; }

# matched vanilla for the EP-off envelope: same MAX_BT=32, no Sluice
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
run "EPoff vanilla_matched(BT32)" /work/ep_van32.log --gpu-memory-utilization 0.90 --max-num-batched-tokens 32

# Sluice, EP off, slots=192 -> envelope 32, WITH full decode graphs
export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=192 SLUICE_ALLOW_FULL_CG=1
run "EPoff SLUICE slots=192   " /work/ep_sl192.log --gpu-memory-utilization 0.90 --max-num-batched-tokens 32
pkill -f 'bin/vllm' 2>/dev/null
echo "EPOFF_DONE" >> "$RES"
