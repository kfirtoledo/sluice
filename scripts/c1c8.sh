#!/bin/bash
# H-KERNEL [10]: V2-Lite and V4, Sluice vs vanilla, at c=1 and c=8.
#
# TWO baselines per model, because they answer different questions:
#   vanilla_stock   - vLLM defaults. What you would actually run WITHOUT Sluice.
#                     This is the honest "should I use this?" comparison.
#   vanilla_matched - vanilla carrying the flags Sluice requires. Isolates
#                     Sluice's OWN cost from the cost of its configuration.
# The campaign's 19.42 ms error came from quoting one and calling it the other.
#
# V4  is on the BREAKABLE path -> SLUICE_ALLOW_FULL_CG=1 keeps the FULL decode
#     graphs (entry [8]: -28%).
# V2-Lite is on the FX-SPLITTING path, where the gap comes from splitting_ops
#     and forcing PIECEWISE is correct -> the entry [8] win should NOT apply.
#     Measured, not assumed: the log line records which mode each arm got.
set -u
export HOME=/work USER=sluice LOGNAME=sluice HF_HOME=/vllm-cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_USE_V2_MODEL_RUNNER=0
RES=/work/c1c8.txt; : > "$RES"
V4=deepseek-ai/DeepSeek-V4-Flash
V4REV=6976c7ff1b30a1b2cb7805021b8ba4684041f136
V2=deepseek-ai/DeepSeek-V2-Lite

serve_v4 () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve $V4 --revision $V4REV --trust-remote-code --port 8000 \
    --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
    --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.55 \
    "$@" >"$LOG" 2>&1 </dev/null & wait_up "$LOG"; }

serve_v2 () { local LOG="$1"; shift
  pkill -f 'bin/vllm' 2>/dev/null; sleep 10
  nohup setsid vllm serve $V2 --trust-remote-code --port 8000 \
    --tensor-parallel-size 1 --moe-backend triton \
    --max-model-len 2048 --gpu-memory-utilization 0.50 \
    "$@" >"$LOG" 2>&1 </dev/null & wait_up "$LOG"; }

wait_up () { local LOG="$1" i
  for i in $(seq 1 75); do
    grep -q 'Application startup complete' "$LOG" 2>/dev/null && return 0
    grep -qiE 'out of memory|Traceback \(most recent|Engine core initialization failed' "$LOG" 2>/dev/null && return 1
    sleep 15
  done; return 1; }

bench () {  # $1 model  $2 tag  $3 concurrency  $4 num-prompts
  local OUT
  OUT=$(timeout 1200 vllm bench serve --base-url http://localhost:8000 \
    --model "$1" --trust-remote-code --dataset-name random \
    --random-input-len 32 --random-output-len 200 --ignore-eos \
    --num-prompts "$4" --max-concurrency "$3" 2>&1 \
    | grep -E 'Mean TPOT|Output token throughput' | tr -s ' ' | tr '\n' ' ')
  echo "$2 c=$3 | $OUT" >> "$RES"
}
both () { bench "$1" "$2" 1 8; bench "$1" "$2" 8 64; }   # warm is c=1's own first run
warm () { bench "$1" "warmup(discard)" 8 16 >/dev/null 2>&1; }

sl_v4 () { export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=48 SLUICE_ALLOW_FULL_CG=1; }
sl_v2 () { export SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_SLOTS=48 SLUICE_ALLOW_FULL_CG=1; }
sl_off () { unset SLUICE_PIECEWISE SLUICE_ROUTER_SPLIT SLUICE_RS_FAST_HIT SLUICE_SLOTS SLUICE_ALLOW_FULL_CG; }
mode_of () { grep -oE "'cudagraph_mode': <CUDAGraphMode\.[A-Z_]+" "$1" | head -1; }

echo "=== DeepSeek-V4-Flash (TP=4, EP, marlin, slots=48) ===" >> "$RES"
sl_off
if serve_v4 /work/c_v4_stock.log; then warm $V4; both $V4 "V4 vanilla_stock  "; mode_of /work/c_v4_stock.log >> "$RES"; else echo "V4 vanilla_stock | FAILED" >> "$RES"; fi
if serve_v4 /work/c_v4_match.log --max-num-batched-tokens 8; then warm $V4; both $V4 "V4 vanilla_matched"; mode_of /work/c_v4_match.log >> "$RES"; else echo "V4 vanilla_matched | FAILED" >> "$RES"; fi
sl_v4
if serve_v4 /work/c_v4_sluice.log --max-num-batched-tokens 8; then warm $V4; both $V4 "V4 SLUICE slots=48"; mode_of /work/c_v4_sluice.log >> "$RES"; grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" /work/c_v4_sluice.log | tail -1 >> "$RES"; else echo "V4 SLUICE | FAILED" >> "$RES"; fi

echo "=== DeepSeek-V2-Lite (TP=1, triton, slots=48) ===" >> "$RES"
sl_off
if serve_v2 /work/c_v2_stock.log; then warm $V2; both $V2 "V2 vanilla_stock  "; mode_of /work/c_v2_stock.log >> "$RES"; else echo "V2 vanilla_stock | FAILED" >> "$RES"; fi
if serve_v2 /work/c_v2_match.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE; then warm $V2; both $V2 "V2 vanilla_matched"; mode_of /work/c_v2_match.log >> "$RES"; else echo "V2 vanilla_matched | FAILED" >> "$RES"; fi
sl_v2
if serve_v2 /work/c_v2_sluice.log --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE; then warm $V2; both $V2 "V2 SLUICE slots=48"; mode_of /work/c_v2_sluice.log >> "$RES"; grep -oE "router-split layers=[0-9]+ gap-calls=[0-9]+ misses=[0-9]+" /work/c_v2_sluice.log | tail -1 >> "$RES"; else echo "V2 SLUICE | FAILED" >> "$RES"; fi

sl_off; pkill -f 'bin/vllm' 2>/dev/null
echo "C1C8_DONE" >> "$RES"
