#!/bin/bash
# H-KERNEL [11]: THE correctness gate. Third attempt, and the first that can work.
#
# Why the first two could not:
#   slots=64 -> every layer static_full -> router-split armed 0/43. No gap to test.
#   slots=62 -> armed, but the cache evicts BETWEEN steps, so misses are
#               unavoidable and BOTH configs legitimately diverge from vanilla.
#               (My "zero misses" prediction was simply wrong.)
#
# SLUICE_RS_FORCE_ARM=1 arms router-split even on static_full layers. Then:
#   every expert resident  -> no misses, no dropped experts
#   router-split ARMED     -> the gap runs every step, under FULL capture
#   => output MUST be BIT-IDENTICAL to vanilla. If FULL capture swallowed the
#      D2H sync and froze the map, this diverges. Nothing else can hide it.
#
# gpu-util 0.40: slots=64 OOMs at 0.55 (it holds every expert AND the KV cache).
# The vanilla reference is regenerated at the SAME gpu-util and MAX_BT so the
# only difference between the arms is Sluice itself.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/gatearm.txt; : > "$RES"
PROMPTS=("The capital of France is" "Explain gravity in one sentence:" "2+2=" "The first three prime numbers are" "Water boils at")

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.40 \
    --max-num-batched-tokens 8 "$@" >"$LOG" 2>&1 </dev/null &
  local i
  for i in $(seq 1 75); do
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
verdict () {  # $1 tag  $2 file  $3 log
  if diff -q /work/gold40.txt "$2" >/dev/null; then
    echo "$1 | *** PASS - BIT-IDENTICAL to vanilla ***" >> "$RES"
  else
    echo "$1 | *** FAIL - diverges from vanilla ***" >> "$RES"
    diff /work/gold40.txt "$2" | head -8 >> "$RES"
  fi
  grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$3" | head -1 >> "$RES"
  grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$3" | tail -1 >> "$RES"
  grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$3" | head -1 >> "$RES"; }

unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG SLUICE_RS_FORCE_ARM
if serve /work/ga_ref.log; then greedy /work/gold40.txt; echo "REF | vanilla captured at gpu-util 0.40" >> "$RES"
else echo "REF | STARTUP FAILED" >> "$RES"; exit 1; fi

export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=64 SLUICE_RS_FORCE_ARM=1

# arm A: the NEW mode. This is the one under test.
export SLUICE_ALLOW_FULL_CG=1
if serve /work/ga_full.log; then greedy /work/ga_full.txt; verdict "GATE_FULL_AND_PIECEWISE" /work/ga_full.txt /work/ga_full.log
else echo "GATE_FULL | STARTUP FAILED: $(grep -oE '(ValueError|torch.OutOfMemoryError): [^\"]{0,90}' /work/ga_full.log | tail -1)" >> "$RES"; fi

# arm B: today's shipping mode. MUST also pass -- if it fails, the gate is
# broken rather than the new mode, exactly as in entries [8] and [9].
export SLUICE_ALLOW_FULL_CG=0
if serve /work/ga_pw.log -cc.cudagraph_mode=PIECEWISE; then greedy /work/ga_pw.txt; verdict "GATE_PIECEWISE(control)" /work/ga_pw.txt /work/ga_pw.log
else echo "GATE_PW | STARTUP FAILED: $(grep -oE '(ValueError|torch.OutOfMemoryError): [^\"]{0,90}' /work/ga_pw.log | tail -1)" >> "$RES"; fi

unset SLUICE_RS_FORCE_ARM; pkill -f 'bin/vllm' 2>/dev/null
echo "GATEARM_DONE" >> "$RES"
