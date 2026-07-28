#!/bin/bash
# H-KERNEL [9]: the correctness gate entry [8] should have had.
#
# Entry [8]'s gate was mis-designed: slots=64 makes every layer static_full, so
# router-split armed 0/43 layers -- a gate with no gap cannot detect a
# swallowed gap. slots=62 fixes it:
#
#   * working set is bounded by the envelope: 8 tokens x top-6 = 48 uniques,
#     so 62 slots => ZERO misses => every routed expert is always resident
#     => output must be BIT-IDENTICAL to vanilla.
#   * 62 < 64 local experts (EP, TP=4), so the layer is NOT static_full and
#     router-split stays ARMED with a live gap.
#
# If FULL capture froze the expert map, this diverges. It is the only
# configuration where "armed" and "deterministic" hold at once.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/gate62.txt; : > "$RES"
PROMPTS=("The capital of France is" "Explain gravity in one sentence:" "2+2=" "The first three prime numbers are" "Water boils at")

serve () {
  local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.55 \
    --max-num-batched-tokens 8 "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 75); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 15
  done
  return 1
}
greedy () {
  local OUT="$1"; : > "$OUT"; local p
  for p in "${PROMPTS[@]}"; do
    curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"deepseek-ai/DeepSeek-V4-Flash\",\"prompt\":\"$p\",\"max_tokens\":24,\"temperature\":0,\"seed\":0}" \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"].replace(chr(10)," "))' >> "$OUT" 2>&1
  done
}
report () {   # $1 tag, $2 file, $3 log
  if diff -q /work/gold.txt "$2" >/dev/null; then
    echo "$1 | PASS - bit-identical to vanilla" >> "$RES"
  else
    echo "$1 | *** DIVERGES from vanilla ***" >> "$RES"
    diff /work/gold.txt "$2" | head -10 >> "$RES"
  fi
  grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$3" | head -1 >> "$RES"
  grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$3" | tail -1 >> "$RES"
  grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$3" | head -1 >> "$RES"
}

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=62

# arm 1: the NEW config -- FULL_AND_PIECEWISE permitted
export SLUICE_ALLOW_FULL_CG=1
if serve /work/g62_full.log; then greedy /work/g62_full.txt; report "GATE62_FULL" /work/g62_full.txt /work/g62_full.log
else echo "GATE62_FULL | STARTUP FAILED" >> "$RES"; fi

# arm 2: today's shipping config -- forced PIECEWISE. Must ALSO pass; if it
# does not, the gate itself is wrong rather than the new mode.
export SLUICE_ALLOW_FULL_CG=0
if serve /work/g62_pw.log -cc.cudagraph_mode=PIECEWISE; then greedy /work/g62_pw.txt; report "GATE62_PIECEWISE" /work/g62_pw.txt /work/g62_pw.log
else echo "GATE62_PIECEWISE | STARTUP FAILED" >> "$RES"; fi

pkill -f 'bin/vllm' 2>/dev/null
echo "GATE62_DONE" >> "$RES"
