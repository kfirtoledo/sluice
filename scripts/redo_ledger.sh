#!/bin/bash
# H-KERNEL [6]: REDO THE LEDGER'S SUBTRACTION, in the ledger's own instrument.
#
# Entry [11] computed the 19.42 ms "graph-shape" term as
#     noop floor (29.25, bench serve) - vanilla (9.83, bench serve)
# with THREE things differing between the arms: the SLUICE_* env, MAX_BT=8, and
# -cc.cudagraph_mode=PIECEWISE. Entry [4] showed the term vanishes under
# `bench latency` at matched flags -- but that is a different instrument, and a
# skeptic should ask whether the instrument, not the flags, is doing the work.
#
# This settles it with bench serve on all three arms, same node, same session:
#
#   A vanilla, ledger flags        -> should reproduce ~9.83 ms
#   B vanilla + MAX_BT=8 + PIECEWISE (NO Sluice) -> the MATCHED baseline
#   C router-split, RS_NOOP        -> should reproduce ~29.25 ms
#
#   B ~ C  => the 19.42 ms is the flags. Entry [4] confirmed, term withdrawn.
#   B ~ A  => the flags are innocent and router-split really does cost ~19 ms
#             under this workload; entry [4] would then be an artifact of
#             `bench latency` and must itself be retracted.
#
# gpu-memory-utilization 0.75 on ALL arms (0.55 OOMs the uncapped arm), so
# absolute numbers may differ slightly from the ledger; the A-B-C contrast is
# what matters and it is internally matched.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
RES=/work/redo.txt; : > "$RES"

run_arm () {
  local TAG="$1"; shift
  local LOG=/work/rd_$TAG.log
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  env "$@" nohup setsid vllm serve deepseek-ai/DeepSeek-V4-Flash \
    --revision $REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.75 \
    ${EXTRA:-} >"$LOG" 2>&1 </dev/null &
  local ok=0 i
  for i in $(seq 1 75); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && { ok=1; break; }
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && break
    sleep 15
  done
  if [ $ok -ne 1 ]; then
    echo "$TAG | STARTUP FAILED: $(grep -oE 'ValueError: [^\"]*' "$LOG" | tail -1 | cut -c1-110)" >> "$RES"
    return
  fi
  local r OUT
  for r in warm r1 r2 r3; do
    OUT=$(timeout 900 vllm bench serve --base-url http://localhost:8000 \
      --model deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
      --dataset-name random --random-input-len 32 --random-output-len 200 \
      --ignore-eos --num-prompts 64 --max-concurrency 8 2>&1 \
      | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
    echo "$TAG $r | $OUT" >> "$RES"
  done
  echo "$TAG | ARMED: $(grep -oE 'router-split layers=[0-9]+' "$LOG" | sort -u | tr '\n' ' ')" >> "$RES"
}

SL="SLUICE_SLOTS=48 SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_RS_NOOP=1"

BUNDLE="--max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE"

EXTRA=""; run_arm A_vanilla_ledgerflags IGNORE=1
EXTRA="$BUNDLE"; run_arm B_vanilla_matched IGNORE=1
EXTRA="$BUNDLE"; run_arm C_routersplit_noop $SL
pkill -f 'bin/vllm' 2>/dev/null
echo "REDO_DONE" >> "$RES"
