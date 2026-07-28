#!/bin/bash
# PDS-PERF H2: decompose V2-Lite's 5.93 ms. Nobody has ever split this number.
#
# V2-Lite c=8 (2026-07-27): vanilla_stock 7.71, vanilla_matched 7.81,
# SLUICE slots=48 13.74. The envelope costs 0.10 ms, so essentially ALL of the
# 5.93 ms gap is Sluice's own machinery. This splits it:
#
#   noop      = SLUICE_RS_NOOP=1 -> streams nothing, syncs nothing, writes
#               nothing. (noop - matched) is the STRUCTURAL term;
#               (sluice - noop) is SYNC + PCIe + map.
#   slots     = 48 / 56 / 62 / 64 -> the PCIe term. V2-Lite runs 24.6% misses
#               at slots=48 (25540/104000); more slots => fewer misses => less
#               PCIe. slots=64 is every expert (static_full), the floor.
#
# PRE-REGISTERED: on V4 the same split gave structural ~0 and streaming ~19 ms.
# I expect the same SHAPE here -- structural near zero, nearly all of the 5.93
# in streaming -- because entry [6b] measured router-split's restructuring at
# 0.06 ms and there is no reason V2-Lite differs. If instead the no-op floor is
# well above matched, something on the fx path costs what the breakable path
# does not, and H1 becomes far more interesting.
# DECISION: whichever term is larger gets the next round of work.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/h2.txt; : > "$RES"
M=deepseek-ai/DeepSeek-V2-Lite

serve () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 8
  nohup setsid vllm serve $M --trust-remote-code --port 8000 \
    --tensor-parallel-size 1 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization 0.50 \
    --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE \
    "$@" >"$LOG" 2>&1 </dev/null &
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
arm () { local TAG="$1" LOG="$2"
  if serve "$LOG"; then
    bench "warm"; sed -i '/^warm |/d' "$RES"
    bench "$TAG r1"; bench "$TAG r2"; bench "$TAG r3"
    { grep -oE "ROUTER-SPLIT armed on [0-9]+/[0-9]+ MoE layers" "$LOG" | head -1
      grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" "$LOG" | tail -1
    } | sed 's/^/      /' >> "$RES"
  else echo "$TAG | FAILED: $(grep -oE '(ValueError|torch.OutOfMemoryError): [^\"]{0,90}' "$LOG" | tail -1)" >> "$RES"; fi; }

SL="SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1"

# control, first
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_RS_NOOP
arm "vanilla_matched (first)" /work/h2_van1.log

# structural / streaming split
export $SL SLUICE_SLOTS=48 SLUICE_RS_NOOP=1
arm "noop floor slots=48   " /work/h2_noop.log
unset SLUICE_RS_NOOP

# PCIe term vs slots
for S in 48 56 62; do
  export $SL SLUICE_SLOTS=$S
  arm "sluice slots=$S       " /work/h2_s$S.log
done

# control, last -- if it disagrees with the first, quote every delta with that band
unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS
arm "vanilla_matched (last) " /work/h2_van2.log

pkill -f 'bin/vllm' 2>/dev/null
echo "H2_DONE" >> "$RES"
