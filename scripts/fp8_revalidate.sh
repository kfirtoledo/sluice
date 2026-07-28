#!/bin/bash
# Re-measure SLUICE_FP8_STREAM on the VALID path.
#
# Commit 5cf32f2 claims +11.3% on V2-Lite at c=8: 13.47 -> 12.04 ms TPOT,
# 528.4 -> 588.1 tok/s, slots=60. That measurement is not usable as it stands:
# valid V2-Lite at c=8 measures ~41 ms (defaults run, 2026-07-28), so 13.47 ms
# is the Inductor+capture regime -- the one whose topk_ids freeze is documented
# by the H9 probes in that same commit. Frozen routing stops fetching experts,
# which is exactly why that regime looks fast and exactly why a *streaming*
# optimisation measured inside it proves nothing.
#
# This re-runs the identical A/B on integration/v025-fp8, where the plugin pins
# VLLM_USE_BREAKABLE_CUDAGRAPH=1 and routing is live.
#
# Design notes:
#   - slots=60 of 64 experts, matching the original claim exactly.
#   - MBT = slots//topk = 60//6 = 10.
#   - A/B/A/B across four server starts, 2 benches each. V2-Lite run-to-run
#     spread reaches 12.7 %, so means alone decide nothing: the bar is
#     NON-OVERLAPPING distributions, the same bar 5cf32f2 set for itself.
#   - fp8 changes the weights, so this arm is VALID_DIVERGENT by construction
#     and gated on coherence, never on bit-identity.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/fp8rev.txt; : > "$RES"
M=deepseek-ai/DeepSeek-V2-Lite
PROMPTS=("The capital of France is" "2+2=" "The first three prime numbers are" "Water boils at")

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve $M --trust-remote-code --moe-backend triton \
    --max-model-len 2048 --max-num-batched-tokens 10 "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 90); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 10
  done; return 1; }

greedy () { local OUT="$1"; : > "$OUT"; local p
  for p in "${PROMPTS[@]}"; do
    curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$M\",\"prompt\":\"$p\",\"max_tokens\":20,\"temperature\":0,\"seed\":0}" \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"].replace(chr(10)," ")[:80])' >> "$OUT" 2>&1
  done; }

coherent () { python3 - "$1" <<'PY'
import sys
bad=0
for line in open(sys.argv[1]):
    w=line.split()
    if len(w)>=8 and len(set(w))<=max(2,len(w)//5): bad+=1
sys.exit(1 if bad else 0)
PY
}

bench () { local OUT
  OUT=$(timeout 1200 vllm bench serve --base-url http://localhost:8000 --model $M \
    --trust-remote-code --dataset-name random --random-input-len 32 \
    --random-output-len 200 --ignore-eos --num-prompts 64 --max-concurrency 8 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "$1 | $OUT" >> "$RES"; }

arm () { local TAG="$1" LOG="$2"
  if ! serve "$LOG"; then
    echo "$TAG | STARTUP FAILED" >> "$RES"
    grep -oE "Sluice: [^\"]{0,140}|(ValueError|RuntimeError|torch.OutOfMemoryError): [^\"]{0,110}" "$LOG" | tail -3 | sed 's/^/      /' >> "$RES"
    return
  fi
  greedy /work/gf_$TAG.txt
  if ! coherent /work/gf_$TAG.txt; then
    echo "$TAG | *** GATE FAIL - degenerate output, NO NUMBERS ***" >> "$RES"
    sed 's/^/      /' /work/gf_$TAG.txt >> "$RES"; return
  fi
  bench "$TAG rep1" ; bench "$TAG rep2"
  { grep -oE "'mode': <CompilationMode\.[A-Z_]+" "$LOG" | head -1
    grep -oE "pinned VLLM_USE_BREAKABLE_CUDAGRAPH=1" "$LOG" | head -1
    grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$LOG" | head -1
    grep -oE "FP8[^\"]{0,80}" "$LOG" | head -2
    grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1
  } | sed 's/^/      /' >> "$RES"; }

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=60

unset SLUICE_FP8_STREAM; arm bf16_A /work/f_bf16a.log
export SLUICE_FP8_STREAM=1; arm fp8_A  /work/f_fp8a.log
unset SLUICE_FP8_STREAM; arm bf16_B /work/f_bf16b.log
export SLUICE_FP8_STREAM=1; arm fp8_B  /work/f_fp8b.log

pkill -f 'bin/vllm' 2>/dev/null
echo "FP8REV_DONE" >> "$RES"
