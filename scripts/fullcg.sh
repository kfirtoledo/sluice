#!/bin/bash
# H-KERNEL [8]: can router-split keep vLLM's FULL decode graphs?
#
# V4's DEFAULT cudagraph_mode is FULL_AND_PIECEWISE. Sluice forces plain
# PIECEWISE, which measured 9.81 -> 25.85 ms/step with NO SLUICE in the process
# (entry [7], arm E). Sluice's guard refuses anything containing "FULL", but
# that guard was written for the fx-splitting path; on V4's BREAKABLE path the
# gap comes from add_eager, not from splitting_ops, so a full graph is still
# segmented at the gap. SLUICE_ALLOW_FULL_CG=1 permits it there.
#
# CORRECTNESS IS GATED FIRST. A wrong answer here is silent: if FULL capture
# did swallow the gap, the expert map would freeze and the model would emit
# plausible-but-wrong tokens. Timing from an arm that fails the gate is
# meaningless and is not reported.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/fullcg.txt; : > "$RES"
PROMPTS=("The capital of France is" "Explain gravity in one sentence:" "2+2=" "The first three prime numbers are" "Water boils at")

serve () {           # serve <log> <extra flags...>  ; env comes from caller
  local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.55 \
    "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 75); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 15
  done
  return 1
}

greedy () {          # greedy <outfile> -- deterministic, temperature 0
  local OUT="$1"; : > "$OUT"
  local p
  for p in "${PROMPTS[@]}"; do
    curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"deepseek-ai/DeepSeek-V4-Flash\",\"prompt\":\"$p\",\"max_tokens\":24,\"temperature\":0,\"seed\":0}" \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"].replace(chr(10)," "))' >> "$OUT" 2>&1
  done
}

bench () {           # bench <tag>
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

SL_BASE="SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1"

# ---- reference: vanilla greedy, default cudagraph mode, SAME MAX_BT ------
# MAX_BT=8 on the reference too: 8-token prefill chunking changes numerics,
# so without it the bit-identity gate would fail for the wrong reason.
if serve /work/fc_ref.log --max-num-batched-tokens 8; then
  greedy /work/gold.txt
  echo "REF | vanilla greedy captured" >> "$RES"
else
  echo "REF | STARTUP FAILED" >> "$RES"; exit 1
fi

# ---- gate 1: FULL residency (slots=64 = local experts under EP/TP=4) -----
# Fully resident => the expert map is static => output MUST match vanilla
# exactly. If FULL capture froze a stale map, this is where it shows.
if env $SL_BASE SLUICE_SLOTS=64 SLUICE_ALLOW_FULL_CG=1 serve /work/fc_gate.log --max-num-batched-tokens 8; then
  greedy /work/gate_full.txt
  if diff -q /work/gold.txt /work/gate_full.txt >/dev/null; then
    echo "GATE1 full-residency + FULL_AND_PIECEWISE | PASS (bit-identical to vanilla)" >> "$RES"
  else
    echo "GATE1 | *** FAIL - output differs from vanilla ***" >> "$RES"
    paste -d'|' /work/gold.txt /work/gate_full.txt >> "$RES"
  fi
else
  echo "GATE1 | STARTUP FAILED (guard refused, or OOM)" >> "$RES"
  grep -oE "Sluice: [^\"]{0,140}" /work/fc_gate.log | tail -2 >> "$RES"
fi

# ---- gate 2 + perf: real offloading, slots=48 ----------------------------
if env $SL_BASE SLUICE_SLOTS=48 SLUICE_ALLOW_FULL_CG=1 serve /work/fc_perf.log --max-num-batched-tokens 8; then
  greedy /work/gate_48.txt
  echo "GATE2 slots=48 greedy output (coherence, NOT expected bit-identical):" >> "$RES"
  sed 's/^/    /' /work/gate_48.txt >> "$RES"
  grep -oE 'permitting cudagraph_mode=[A-Z_]+' /work/fc_perf.log | head -1 >> "$RES"
  bench "PERF_fullcg_slots48"
else
  echo "PERF | STARTUP FAILED" >> "$RES"
fi

# ---- control: same thing but forced PIECEWISE (today's config) -----------
if env $SL_BASE SLUICE_SLOTS=48 serve /work/fc_ctrl.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE; then
  bench "CTRL_piecewise_slots48"
else
  echo "CTRL | STARTUP FAILED" >> "$RES"
fi
pkill -f 'bin/vllm' 2>/dev/null
echo "FULLCG_DONE" >> "$RES"
