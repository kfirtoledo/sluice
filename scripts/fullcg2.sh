#!/bin/bash
# H-KERNEL [8]: can router-split keep vLLM's FULL decode graphs?  (retry)
#
# Fix vs the first attempt: `env VAR=v serve ...` cannot invoke `serve`, which
# is a shell FUNCTION -- env only execs real binaries ("env: 'serve': No such
# file or directory"). Sluice vars are now exported/unset around the calls.
# The vanilla reference (/work/gold.txt) survived the first run and is reused.
#
# Entry [7]: forcing PIECEWISE costs +12.13 ms marginal over MAX_BT=8 alone,
# with no Sluice in the process. V4's default is FULL_AND_PIECEWISE. Sluice
# refuses anything containing "FULL", but that guard targets the fx-splitting
# path; on V4's breakable path the gap comes from add_eager regardless of
# cudagraph_mode. SLUICE_ALLOW_FULL_CG=1 permits it only there.
#
# CORRECTNESS IS GATED FIRST -- a swallowed gap freezes the expert map and
# yields fluent but WRONG tokens, with no exception raised.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/fullcg.txt; : > "$RES"
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
  local OUT="$1"; : > "$OUT"
  local p
  for p in "${PROMPTS[@]}"; do
    curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"deepseek-ai/DeepSeek-V4-Flash\",\"prompt\":\"$p\",\"max_tokens\":24,\"temperature\":0,\"seed\":0}" \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"].replace(chr(10)," "))' >> "$OUT" 2>&1
  done
}

bench () {
  local TAG="$1" r OUT
  for r in warm r1 r2 r3; do
    OUT=$(timeout 900 vllm bench serve --base-url http://localhost:8000 \
      --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
      --dataset-name random --random-input-len 32 --random-output-len 200 \
      --ignore-eos --num-prompts 64 --max-concurrency 8 2>&1 \
      | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
    echo "$TAG $r | $OUT" >> "$RES"
  done
}

sluice_on () {   # $1 = slots ; $2 = 1 to allow FULL cudagraph
  export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1
  export SLUICE_SLOTS="$1" SLUICE_ALLOW_FULL_CG="$2"
}
sluice_off () {
  unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT \
        SLUICE_SLOTS SLUICE_ALLOW_FULL_CG
}

echo "REF | reusing /work/gold.txt from the previous run (vanilla, MAX_BT=8, default cudagraph)" >> "$RES"

# ---- gate 1: FULL residency (slots=64 = local experts under EP/TP=4) ------
# Fully resident => static expert map => output MUST match vanilla exactly.
sluice_on 64 1
if serve /work/fc_gate.log; then
  greedy /work/gate_full.txt
  if diff -q /work/gold.txt /work/gate_full.txt >/dev/null; then
    echo "GATE1 slots=64 + FULL_AND_PIECEWISE | PASS (bit-identical to vanilla)" >> "$RES"
  else
    echo "GATE1 | *** FAIL - output differs from vanilla ***" >> "$RES"
    diff /work/gold.txt /work/gate_full.txt | head -12 >> "$RES"
  fi
  grep -oE 'permitting cudagraph_mode=[A-Z_]+' /work/fc_gate.log | head -1 >> "$RES"
  grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" /work/fc_gate.log | head -1 >> "$RES"
else
  echo "GATE1 | STARTUP FAILED" >> "$RES"
  grep -oE "Sluice: [^\"]{0,150}" /work/fc_gate.log | tail -2 >> "$RES"
  grep -oE "(ValueError|RuntimeError|torch.OutOfMemoryError): [^\"]{0,110}" /work/fc_gate.log | tail -1 >> "$RES"
fi

# ---- gate 2 + perf: real offloading, slots=48, FULL_AND_PIECEWISE --------
sluice_on 48 1
if serve /work/fc_perf.log; then
  greedy /work/gate_48.txt
  echo "GATE2 slots=48 greedy (coherence; NOT expected bit-identical):" >> "$RES"
  sed 's/^/    /' /work/gate_48.txt >> "$RES"
  grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" /work/fc_perf.log | head -1 >> "$RES"
  bench "PERF_fullcg_slots48"
else
  echo "PERF | STARTUP FAILED" >> "$RES"
  grep -oE "(ValueError|RuntimeError|torch.OutOfMemoryError): [^\"]{0,110}" /work/fc_perf.log | tail -1 >> "$RES"
fi

# ---- control: identical, but forced PIECEWISE (today's shipping config) ---
sluice_on 48 0
if serve /work/fc_ctrl.log -cc.cudagraph_mode=PIECEWISE; then
  greedy /work/ctrl_48.txt
  echo "CTRL slots=48 greedy:" >> "$RES"
  sed 's/^/    /' /work/ctrl_48.txt >> "$RES"
  bench "CTRL_piecewise_slots48"
else
  echo "CTRL | STARTUP FAILED" >> "$RES"
fi
sluice_off
pkill -f 'bin/vllm' 2>/dev/null
echo "FULLCG_DONE" >> "$RES"
