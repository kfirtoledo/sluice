#!/bin/bash
# H-KERNEL [11]: THE correctness gate, at TP=8 where it finally fits.
#
# Three failed attempts before this, each recorded:
#   slots=64, TP=4          -> every layer static_full, router-split armed 0/43.
#                              A gate with no gap cannot detect a swallowed gap.
#   slots=62, TP=4          -> armed, but the cache evicts BETWEEN steps so
#                              misses are unavoidable; BOTH configs legitimately
#                              diverge from vanilla. My "zero misses" was wrong.
#   slots=64 + FORCE_ARM, TP=4 -> OOM: slot buffers are 45.9 GiB ON TOP of the
#                              loaded weights. Infeasible in 80 GiB at any
#                              gpu-memory-utilization.
#
# TP=8 fixes the memory: with EP each rank holds 256/8 = 32 experts per layer,
# so slots=32 IS full residency and the buffers are ~23 GiB against ~19 GiB of
# weights. Same BREAKABLE path, same FULL capture -- the thing under test is
# unchanged.
#
# With every expert resident and SLUICE_RS_FORCE_ARM=1:
#   no misses, no dropped experts, router-split ARMED, gap runs every step
#   => output MUST be BIT-IDENTICAL to vanilla.
# If FULL capture froze the expert map, this is where it shows.
#
# envelope: MAX_BT <= slots//top_k = 32//6 = 5.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/gate_tp8.txt; : > "$RES"
PROMPTS=("The capital of France is" "Explain gravity in one sentence:" "2+2=" "The first three prime numbers are" "Water boils at")

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 12
  nohup setsid vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 8 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.75 \
    --max-num-batched-tokens 5 "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 90); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 15
  done; return 1; }
greedy () { local OUT="$1"; : > "$OUT"; local p
  for p in "${PROMPTS[@]}"; do
    curl -s http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"deepseek-ai/DeepSeek-V4-Flash\",\"prompt\":\"$p\",\"max_tokens\":24,\"temperature\":0,\"seed\":0}" \
      | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["text"].replace(chr(10)," "))' >> "$OUT" 2>&1
  done; }
verdict () {
  if diff -q /work/gold8.txt "$2" >/dev/null; then
    echo "$1 | *** PASS - BIT-IDENTICAL to vanilla ***" >> "$RES"
  else
    echo "$1 | *** FAIL - diverges from vanilla ***" >> "$RES"
    diff /work/gold8.txt "$2" | head -8 >> "$RES"
  fi
  grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$3" | head -1 >> "$RES"
  grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$3" | tail -1 >> "$RES"
  grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$3" | head -1 >> "$RES"; }

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG SLUICE_RS_FORCE_ARM
if serve /work/t8_ref.log; then greedy /work/gold8.txt; echo "REF | vanilla TP=8 captured" >> "$RES"; sed 's/^/    /' /work/gold8.txt >> "$RES"
else echo "REF | STARTUP FAILED: $(grep -oE '(ValueError|torch.OutOfMemoryError): [^\"]{0,90}' /work/t8_ref.log | tail -1)" >> "$RES"; exit 1; fi

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=32 SLUICE_RS_FORCE_ARM=1

export SLUICE_ALLOW_FULL_CG=1              # the mode under test
if serve /work/t8_full.log; then greedy /work/t8_full.txt; verdict "GATE_FULL_AND_PIECEWISE" /work/t8_full.txt /work/t8_full.log
else echo "GATE_FULL | STARTUP FAILED: $(grep -oE '(ValueError|torch.OutOfMemoryError): [^\"]{0,90}' /work/t8_full.log | tail -1)" >> "$RES"; fi

export SLUICE_ALLOW_FULL_CG=0              # control: must ALSO pass
if serve /work/t8_pw.log -cc.cudagraph_mode=PIECEWISE; then greedy /work/t8_pw.txt; verdict "GATE_PIECEWISE(control)" /work/t8_pw.txt /work/t8_pw.log
else echo "GATE_PW | STARTUP FAILED: $(grep -oE '(ValueError|torch.OutOfMemoryError): [^\"]{0,90}' /work/t8_pw.log | tail -1)" >> "$RES"; fi

unset SLUICE_RS_FORCE_ARM; pkill -f 'bin/vllm' 2>/dev/null
echo "GATE_TP8_DONE" >> "$RES"
