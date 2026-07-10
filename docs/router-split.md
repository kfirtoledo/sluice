# Router-split: CUDA-graph decode under expert offload

`SLUICE_ROUTER_SPLIT=1` restructures each MoE layer's compiled graph so that
everything except a thin streaming gap is captured by vLLM's own piecewise
CUDA-graph machinery — while Sluice keeps offloading experts to host RAM and
streaming router-selected experts into GPU slot caches every step.

Companion flags: `SLUICE_PIECEWISE=1` (required) and `SLUICE_RS_FAST_HIT=1`
(exact all-hit fast path). Shorthand used throughout this doc:
`INDUCTOR_PARTITION=1` stands for engine-side
`use_inductor_graph_partition=true`, and `MAX_BT=<n>` for
`--max-num-batched-tokens <n>` (set to `slots//topk`).
Diagnostic only: `SLUICE_RS_NOOP=1` (outputs invalid by design).

Headline (details in [§5](#5-performance-results-and-mechanisms)): V2-Lite
single-stream decode 27 → 147 tok/s (5.4× the previous plugin floor, 55% of
the vanilla full-graph ceiling); V2-Lite c=8 parity-to-+8% **over** vanilla
full graphs while offloading a quarter of the experts; Qwen3-30B c=8 +16%
over the fully-resident baseline with half the experts in host RAM, and +80%
at matched c=12.

Sources: all measurements are from the July 2026 campaign (§5 carries the
numbers, §1 the eager-hook prehistory, §6 the eviction-policy null under
router-split), and the implementation lives in
[`src/sluice/offloader.py`](../src/sluice/offloader.py). Background on the
offloader itself: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## 1. Problem: offload forbade CUDA graphs — a 10× decode gap

Sluice's classic mechanism wraps each MoE layer's `quant_method.apply` with a
per-layer hook that syncs the router's `topk_ids` to host, streams missing
experts into slots, and rewrites the layer's `expert_map`. A CUDA graph
cannot contain that hook: capture would freeze the expert map and replay it
with stale routing — silently wrong outputs. Sluice therefore refuses FULL
graphs at config time and historically forced `enforce_eager`.

The cost of eager is not the hook tax (~6%, within run-to-run noise); it is
forfeiting graphs entirely. On DeepSeek-V2-Lite (one node, one job):

| arm | c=1 | c=8 |
|---|---|---|
| stock vLLM, eager | 25.2 | 202.7 |
| **stock vLLM, CUDA graphs** | **265.8** | **982.9** |
| Sluice hook, eager | 25.5 | 191.5 |

Stock vLLM decode runs 10.4× above the offloaded eager floor at c=1. Two
partial escapes preceded router-split:

- **v3 static-full graphs**: with the full
  working set resident and the map constant, FULL capture is safe —
  bit-identical, 982.2 vs 982.9 tok/s. But it requires full residency, i.e.
  no offload.
- **Classic piecewise** (`SLUICE_PIECEWISE=1`): appending
  `vllm::moe_forward` / `vllm::moe_forward_shared` to
  `compilation_config.splitting_ops` makes the fx splitter cut the compiled
  graph at every MoE, so attention/norms are captured and the whole MoE +
  hook runs in an eager gap. Offload and graphs coexist for the first time
  — but the entire MoE (gate, select, GEMM, shared experts, hook) stays
  eager. Best classic-piecewise result: 27 tok/s c=1, ~205 c=8 at slots=48
  — still ~10× below the vanilla c=1 ceiling.

The residual is structural: the expensive, graph-friendly work (the expert
GEMM, the shared experts) is trapped in the eager gap only because the tiny
streaming hook lives in the middle of it. Router-split extracts the hook.

## 2. Design

### 2.1 The traced-entry restructure

vLLM v0.23 dispatches the entire MoE layer through the opaque custom ops
`vllm::moe_forward[_shared]`; dynamo never traces inside them, which is why
the classic hook is invisible to compile and the whole layer must be a gap.
But the runner exposes a patchable seam **above** the opaque op:
`runner._forward_entry` — an attribute holding the op handle, called from
traced Python (`MoERunner.forward`). Replacing that attribute with a plain
Python function restructures what dynamo traces without forking vLLM.

The replacement entry (see `attach_router_split` in
[`src/sluice/offloader.py`](../src/sluice/offloader.py)):

```
def _forward_entry(hs, rl, sei, iid, lname, unpad):
    if hs.shape[0] > thr:                       # envelope branch (§3)
        return orig(hs, rl, sei, iid, lname, unpad)
    if gate is not None:
        rl, _ = gate(hs)                        # traced, captured
    tw, ti = select_experts(hs, rl, iid)        # traced, captured
    h  = torch.ops.vllm.sluice_stream_gap(hs, ti, idx)   # the ONLY eager gap
    routed = torch.ops.vllm.fused_experts(      # traced, captured
        h, w13, w2, tw, ti, activation="silu",
        global_num_experts=gne, expert_map=emap)
    sh = shared_layer(sei if sei is not None else hs)    # traced, captured
    return (sh, routed)
```

Layer by layer:

- **Gate linear, traced.** Stock `_forward_impl` *overwrites* incoming
  `router_logits` with `self.gate(hidden)` when the runner holds the gate
  (all 26 V2-Lite layers do). The traced entry must mirror that — a plain
  traceable linear — not skip it. (A first version guarded on runner-held
  gates and armed 0/26 layers; this was null-result #1, diagnosed from the
  arm counter.)
- **`select_experts`, traced** — the first time dynamo ever sees routing.
  Under the classic hook it was buried inside the opaque op.
- **`vllm::sluice_stream_gap`** — the only eager gap (§2.2).
- **`torch.ops.vllm.fused_experts`, traced** — the expert GEMM, called
  directly with `expert_map=` pointing at Sluice's per-layer map buffer. It
  lands in a captured piece; vLLM's own capture machinery records it,
  including the shared-experts aux-stream overlap that made private capture
  impossible (§6, gemm-graph).
- **Shared experts, traced** — likewise captured by vLLM itself.
- **The stock traced tail** (combine / scale / reduce after
  `_forward_entry` returns) is untouched — fidelity there is inherited, not
  reimplemented.

Unsupported layers fail closed to the stock path: the patcher skips
`static_full` layers, monolithic quant methods (e.g. marlin), fused-gate
variants, naive dispatch/combine, and `pcp_size > 1`, and logs the skip
census. Prefill-sized steps outside the envelope keep the classic
wave-capable hook.

### 2.2 The gap op

`vllm::sluice_stream_gap(hidden_states, topk_ids, layer_idx)` is registered
via vLLM's `direct_register_custom_op` (with a fake impl for tracing) and
appended to `compilation_config.splitting_ops`, so the splitter cuts the
graph exactly there. Its body is the minimum that cannot be captured:

1. one D2H copy of the tiny `topk_ids` tensor (plus its sync),
2. stream this step's missing experts into LRU slots,
3. refresh the standing `expert_map` contents.

To the graph the op is functionally pure: it returns
`hidden_states.clone()`, which `fused_experts` consumes — ordering is
enforced by dataflow, so no compiler pass can hoist the GEMM above the
stream. `mutates_args=[]`; the real side effects are content writes to
buffers the graph reads by pointer (§2.3).

With `SLUICE_RS_FAST_HIT=1`, all-hit gaps (no missing expert) skip the
unique/LRU-touch/map-check bookkeeping. The D2H sync stays — it is the
exactness anchor; the only consequence of skipped recency touches is victim
choice on a later miss, never a wrong map. 86–99% of gaps take this path in
steady state.

### 2.3 Pointers vs. contents

The correctness kernel of the whole design: **captured pieces bake pointers;
the gap rewrites contents.**

- The capture records the *addresses* of the slot weight buffers
  (`w13_weight`, `w2_weight`) and of the `expert_map` buffer — all allocated
  once and never reallocated.
- Before every replay reads them, the eager gap has already rewritten their
  *contents* (streamed weights, updated map) on the same stream.

The classic-piecewise structural argument was "captured pieces contain no
expert-map reads." Router-split deliberately moves a map read *into* a
captured piece, and replaces the argument with this stronger one: the read
is by pointer, and the gap's dataflow position guarantees the pointed-to
contents are current. All five private-capture hazards (§6, gemm-graph)
vanish structurally, because vLLM's own capture — which knows its streams,
its warmup passes, and its buffer lifetimes — records the GEMM.

### 2.4 Why a plugin, not a fork — the exact vLLM seams

Every mechanism rides a stable seam; no vLLM line is edited. The full
integration surface:

| seam | use |
|---|---|
| `vllm.general_plugins` entry point (`sluice = "sluice.plugin:register"`, [`pyproject.toml`](../pyproject.toml)) | loads Sluice in every engine/worker process at startup |
| `compilation_config.splitting_ops` | appended at offloader construction, before compile: `vllm::moe_forward`, `vllm::moe_forward_shared`, and `vllm::sluice_stream_gap` |
| `runner._forward_entry` on `MoERunner` | the patch point: an attribute called from traced Python, normally the opaque op handle |
| `runner.gate`, `runner.router.select_experts` | stock routing components invoked from the traced entry, mirroring stock `_forward_impl` |
| `torch.ops.vllm.fused_experts` | vLLM's registered fused-MoE op, called directly with Sluice's `expert_map` buffer |
| `runner._shared_experts._layer` | the shared-experts module, invoked traced |
| `vllm.utils.torch_utils.direct_register_custom_op` | registers the gap op with a fake impl |
| `scheduler_config.max_num_batched_tokens` | read for the config-time envelope assertion (§3) |
| `quant_method.apply` monkeypatch | the pre-existing classic hook, kept as the fail-closed fallback path |

A fork could do better in specific ways (§7), but nothing in the mechanism
*requires* one: the restructure is a few hundred lines of patching against
attributes vLLM already exposes, installable as a pip package, and it fails
closed to the classic path wherever a model shape is out of scope.

## 3. The envelope

Two facts about vLLM's compile pipeline shape the deployment contract:

1. **Dynamo traces the model ONCE**, with the profile run's batch size
   (`max_num_batched_tokens`) as the size hint, and vLLM's custom dispatcher
   never re-evaluates guards. A Python branch like
   `if tokens > thr: classic else: rsplit` is resolved *at trace time* and
   burned into the artifact. With the default 512-token hint the trace took
   the classic side, and the rsplit path simply did not exist in the
   compiled model — no crash, gap-calls=0, arms identical (null-result #2,
   diagnosed from the engagement counters).
2. **The gap is single-wave by contract**: a captured `fused_experts` piece
   replays once per step, so every step's unique-expert demand
   (`tokens × topk`) must fit the slot cache.

Both are solved by bounding the batch envelope:

```
max_num_batched_tokens <= SLUICE_SLOTS // topk        (thr)
```

asserted loudly at config time (a violating config raises with the exact
`--max-num-batched-tokens` to set). Consequences:

- The trace hint lands on the rsplit side, so the compiled artifact contains
  the split path — and *only* needs it, since no step can exceed `thr`.
- Every step satisfies the single-wave contract *by construction*; overflow
  is a loud error, never a wrong answer (zero overflows observed across all
  campaigns).
- vLLM captures pieces per batch size within the envelope (per-size
  capture), so decode steps replay size-appropriate graphs; the capture
  ladder never extends past `thr`.
- **Prefill runs in `thr`-token chunks** (slow); decode is the product. This
  is a decode-worker envelope — the same P/D-disaggregation regime the hook
  work already targeted.
- The envelope itself is nearly free: a `base8` control arm (same
  `MAX_BT=8`, no rsplit) measured within ~4–6% of the `base512` classic
  floor, so none of the rsplit win is envelope artifact.

Examples: V2-Lite (top-6) slots=48 → `MAX_BT=8`; Qwen3-30B (top-8) slots=96
→ 12, slots=64 → 8, slots=48 → 6. In pure eager mode (no piecewise) the
branch is evaluated per step instead, and oversized steps fall back to the
classic hook — no envelope needed.

## 4. Correctness methodology

### 4.1 Bit-compare gates

All gates compare greedy token ids byte-for-byte (md5 over outputs), stock
hook vs router-split, same weights, same node, same job.

| gate | setup | result |
|---|---|---|
| Gate A — eager fidelity | slots=48, MAX_SEQS=1, compile off | **bit-identical**; 26/26 layers armed, gap-calls=8,000. The traced gate+select+fused_experts path reproduces stock kernels exactly. |
| Forced-miss gate | slots=36 < decode working set, eager | **bit-identical with 11,497 misses** streamed through the gap |
| Fast-hit variant | same forced-miss setup, `SLUICE_RS_FAST_HIT=1` | **bit-identical** |
| Multi-token chunks | chunked prefill inside the envelope (multi-token steps through the gap), with live streaming | **bit-identical under ~6,800 live misses** |

The multi-token gate matters because under the piecewise envelope *prefill
chunks also take the rsplit path* — the gap must be exact for token counts
in `[1, thr]`, not just single-token decode.

### 4.2 Engagement counters

Null-result #2 (§3) established the rule: **a passing bit-gate with zero
engagement proves nothing** — the path may not exist in the compiled
artifact. Every run therefore self-reports from inside the gap
(`DIAG[rs]` every 2,000 gap calls): layers armed (must be 26/26 on V2-Lite),
gap-calls, misses, fast-hits, stage-falls, and a loud `NOOP-GAP (OUTPUTS
INVALID)` banner when the diagnostic arm is active. The patcher additionally
logs a skip census (`static-full`, `monolithic`, `fused-gate`, …) so a
partially-armed model is visible at startup. Perf claims are only accepted
alongside engagement numbers (e.g. the headline arms: gap-calls=14,000,
zero single-wave overflows; the miss-path arm: 9,114 misses while running
125.5 tok/s).

### 4.3 Fidelity ledger across models

- **V2-Lite: bit-exact everywhere** (all gates above).
- **Qwen3-30B: eager rsplit-vs-classic diverges — deterministically.**
  Forensics: classic==classic, nofh==nofh, fh==fh across repeated runs (each
  arm bit-reproducible); fh, nofh, classic mutually differ; every output
  coherent. `SLUICE_RS_DUMP=1` confirms an identical routing config is
  consumed (RenormalizeNaive, silu, gne=128). Verdict: deterministic fp
  differences between kernel stacks — classic `apply` routes through the
  modular prepare/finalize stack, rsplit calls monolithic `fused_experts` —
  plus placement-sensitive accumulation on this shape. Same legitimacy
  class as vLLM's own eager-vs-compiled numeric drift; not a logic bug;
  quality unaffected.

Piecewise arms are additionally covered by the structural argument of §2.3;
a formal transparency smoke (sluice-piecewise vs vanilla-piecewise on a
resident model) is queued in §7.

### 4.4 Red-team

A dedicated adversarial pass probed seven threats against the running
system. **Zero Sluice defects.** The one finding was upstream.

| # | threat | probe | outcome |
|---|---|---|---|
| 1 | miss path corrupts state under real streaming | forced-miss bit-gate, slots < working set | bit-identical (11,497 misses) |
| 2 | multi-token chunks break the gap (prefill inside the envelope) | mixed-chunk bit-gate with live streaming | bit-identical (~6,800 misses) |
| 3 | fast-hit recency staleness corrupts the map | fast-hit forced-miss bit-gate + invariant review | bit-identical; staleness can only shift victim choice, never map correctness |
| 4 | vLLM capture/warmup passes enter the gap (the hazard class that killed private capture) | engagement counters through capture passes; the staged arm's stage-falls matched capture passes exactly (156/288) | no corruption; passes detected exactly |
| 5 | single-wave overflow (batch shaped past `slots//topk`) | config-time assert + runtime overflow check | refused loudly at config; zero overflows across all campaigns; never a wrong answer |
| 6 | compiler drops or reorders the pure-to-graph gap op | dataflow-ordering review (gap output feeds `fused_experts`) + post-compile engagement | gap present and ordered; gap-calls as expected |
| 7 | sampling-feature interactions | feature sweep incl. `prompt_logprobs` | **NaN artifact found — and it reproduces on vanilla vLLM under piecewise, without Sluice**: a pre-existing upstream artifact of `prompt_logprobs` + piecewise compilation, not a router-split defect |

## 5. Performance results and mechanisms

All numbers: one node, one job per bracket, decode-isolated lockstep ladder,
measured in the July 2026 campaign.

### 5.1 V2-Lite (26 MoE layers × 64 experts top-6, 1×H100)

The cumulative single-stream arc, c=1 decode tok/s:

```
eager-offload 24 → piecewise 26 → +slots/hook-lite 27   (the old plugin floor)
→ router-split 112 → +fast-hit 136 → +inductor partition 147
```

5.4× the old floor; 55% of the vanilla full-graph ceiling (267). At c=8:
205 → 1063, **past the vanilla full-graph baseline (985)**.

Key brackets (slots=48 → 16/64 experts offloaded, `MAX_BT=8`):

| arm | c=1 | c=8 |
|---|---|---|
| base512 (classic piecewise floor) | 26.0 (38.4 ms) | 194.6 |
| base8 (envelope only, no rsplit) | 25.0 | 182.7 |
| rsplit8 | 112.2 (8.9 ms) | 599.0 |
| rsplit + fast-hit | 135.7 (7.37 ms) | 639.7 |
| **rsplit + fast-hit + inductor partition** | **147.2 (6.79 ms)** | **1062.6 (7.53 ms)** |
| vanilla FULL-graph baseline | 267.1 (3.7 ms) | 984.9 (8.1 ms) |

- 4.5× single-stream and 3.3× at c=8 over the classic-hook floor from the
  restructure alone; the base8 control shows the envelope contributes none
  of it (it is a small cost, not a win).
- Fast-hit: +16% c=1 (116.6 → 135.7 on its own bracket); 86% of gaps take it.
- Inductor graph partition (`use_inductor_graph_partition=true`): partitions
  at inductor level instead of fx — cheaper piece transitions and fusion
  across piece boundaries. +8.5% c=1, +66% c=8.
- The c=8 crossing replicated on a heavier workload in the eviction-policy
  matrix: 990–1066 across four arms vs vanilla 985 — the honest claim is
  **parity-to-+8%** at c=8 while offloading a quarter of the experts.
- Miss path is fast, not just correct: at slots=36 (< working set,
  streaming every ~1.5 gaps) rsplit holds 125.5 tok/s c=1 with 9,114
  misses — 5.1× the classic hook at equal slots.

**Single-stream floor decomposition** (`SLUICE_RS_NOOP=1` gap; outputs
invalid, structure-only): vanilla FULL 3.7 ms < noop floor 6.51 ms <
fast-hit 7.37 ms < full gap 8.57 ms (fx splitter). Reading: structural
split tax ~2.8 ms (dominant), D2H sync ~0.9 ms, LRU bookkeeping ~1.2 ms
(reclaimed by fast-hit). Under inductor partition the noop floor drops to
5.36 ms — leaving ~1.4 ms of gap work and ~1.7 ms of piece-dispatch
structure between the shipped config (6.79 ms) and vanilla (3.7 ms). The
residual is vLLM-side structure, not plugin work (§7).

### 5.2 Qwen3-30B-A3B (48 MoE layers × 128 experts top-8, 1×H100)

Fits resident on one H100 (~61 GB), so a true vanilla FULL-graph baseline
exists. rsplit offloads **half** the experts (slots=64, ~29 GB freed for
KV) with continuous streaming:

| arm | c=1 | c=8 |
|---|---|---|
| vanilla resident + FULL graphs | 210.1 (4.76 ms) | 742.3 (10.78 ms) |
| rsplit(+fh) slots=64 | 120.6 (8.29 ms) | 546.1 |
| **rsplit(+fh) + inductor partition** | 121.2 (8.25 ms) | **864.5 (9.25 ms)** |

**c=8: 864.5 = 116% of the resident baseline — with half the experts in
host RAM.** Two scale observations: partition gave +58% at c=8 on 48 layers
(vs +66% on 26-layer V2-Lite) but ~0 at c=1, where 48 per-layer D2H syncs
(~70 µs each) bound the single stream. And **classic piecewise does not
work at all on this model** — putting the moe ops in `splitting_ops` hits a
vLLM codegen `AssertionError` (`codegen.py
generate_execution_code_with_name`) at any `max_bt` — so router-split is
the *only* offload+graphs path on Qwen3, not merely the fastest.

**Offload-fraction sweep** (rsplit+fh+partition):

| slots (offloaded) | freed HBM | c=1 | aggregate |
|---|---|---|---|
| 96 (25%) | ~11 GB | 122.8 | 883.7 @c=8 / **1263.8 @c=12** |
| 64 (50%) | ~29 GB | 121.2 | 864.5 @c=8 |
| 48 (62%) | ~37 GB | 119.2 | 645.6 @c=6 (envelope cap) |
| vanilla resident | — | 210.1 | 742.3 @c=8; 700.7 @c=12 |

**c=1 is flat across the offload range (~1% per tier)** — under
router-split the offload fraction is nearly free in single-stream latency.
The earlier D-KV failure mode (TPOT proportional to offload fraction,
canceling the freed-KV concurrency win) is structurally gone. Aggregate
scales with the envelope (`MAX_BT = slots//topk`). Fast-hit rate 99% in all
sweep arms; ~1.2–1.5k misses per arm (steady streaming).

### 5.3 Mechanisms

Three distinct mechanisms produce the wins; they are worth separating
because they predict where the approach does and does not transfer.

1. **Kernel-stack difference.** The restructure moves gate, `select_experts`
   and the expert GEMM from an opaque eager op into torch.compile's world:
   inductor fuses across them, and piece launches replace launch-by-launch
   eager dispatch. The classic path additionally routes through the modular
   prepare/finalize MoE stack, while rsplit calls monolithic
   `fused_experts` directly (confirmed by `RS-DUMP` forensics on Qwen3 —
   also the source of that model's benign fp divergence, §4.3). This is the
   bulk of the 4.5× single-stream jump.

2. **Compile-range specialization.** The envelope bounds every traced and
   captured shape at `thr` tokens, so the single dynamo trace and all
   captured pieces specialize for decode-sized batches instead of a generic
   512-token hint. The base8 control shows the envelope alone is worth
   nothing (slightly negative) — the value is the specialization the
   *compiler* extracts from the bounded range, which inductor partition
   then compounds (fusion across piece boundaries; +58–66% at c=8).

3. **Capture-size padding at the unique-expert bandwidth wall** — how an
   offloaded model *beats* its resident baseline. MoE decode weight traffic
   scales with *unique experts touched* per layer, ≈ `min(tokens × topk,
   N)`, not with tokens. Vanilla's FULL-graph capture ladder jumps 8 → 16,
   so a c=12 batch is padded to 16 rows: on Qwen3 (top-8, N=128) true c=12
   touches ≤ 96 unique experts per layer, while the padded 16-row replay
   touches up to all 128 — ~33% more weight reads for zero useful tokens,
   at the bandwidth wall. Measured: vanilla TPOT collapses 10.78 → 17.13 ms
   from c=8 to c=12 (aggregate *drops* 742 → 701), while the rsplit
   envelope at slots=96 (`MAX_BT=12`) never pads past the true batch and
   holds 9.5 ms — **1263.8 vs 700.7 tok/s, +80% at matched c=12.** The
   same effect explains the V2-Lite c=8 crossing (+8%): the offloaded
   config's compiled decode step simply does less redundant work per token
   than vanilla's padded FULL-graph replay.

### 5.4 Memory-budget rule

Slot buffers allocate **after** vLLM's memory profiling, so they must be
budgeted **outside** `gpu_memory_utilization` — a violated budget shows up
as OOM at first allocation, not at profile time. Confirmed repeatedly on
80 GB H100: Qwen3 slots=64 → `GPU_MEM=0.55`; slots=96 (~43 GB of slots) →
`0.40`; slots=48 → `0.45`.

## 6. Negative results (kept in-tree, OFF, with warnings)

Each was built, gated for correctness, measured, and turned off. The
post-mortems are the reusable knowledge.

- **`SLUICE_RS_STAGED` — staged-ids doorbell (−6–8%, both models).**
  Capturable async D2H of `topk_ids` into pinned per-layer buffers inside
  the captured piece, doorbell-stamped; the gap spin-waits the doorbell
  instead of syncing the stream (classic-sync fallback self-heals capture
  passes). Correct and engaged (bit-identical under forced streaming;
  stage-falls == capture passes exactly). Post-mortem: under inductor
  partition the per-gap sync it replaces is already a ~32-byte copy on an
  idle DMA engine — three extra captured ops per layer plus spin/resync
  bookkeeping cost more than the sync they save.

- **`SLUICE_LAZY_STEP` — one-sync-per-step (−12–17%).** Run the forward
  optimistically sync-free against standing maps with a device miss
  counter, one boundary check, exact same-step classic rerun on miss.
  Bit-identical, 77% clean forwards at steady state — and slower.
  Post-mortem: piecewise already demoted the per-layer syncs (the captured
  pieces finish fast and the CPU runs ahead), so the savings evaporated
  while the costs (per-layer miss-check launches every clean forward, 2×
  rerun on ~23% of steps) remained. Untested hypothesis: may still win in
  pure-eager worlds (e.g. a 61-layer V4 eager path) where syncs genuinely
  are the wall.

- **`SLUICE_GEMM_GRAPH` — private per-layer capture (PARKED).** Five
  distinct capture hazards found by the bit-compare gate across five
  forensic rounds: (1) the moe op's `mutates_args` contract requires
  copying mutated static buffers back on replay; (2) per-step tensors must
  be staged by object identity, never shape; (3) capture must never land
  inside the host engine's own capture/profiling passes; (4) the
  unfixable-as-a-plugin one — `apply()` runs shared experts with
  aux-stream overlap invisible to a single-stream private capture; (5) a
  fifth hidden-state hazard at the `_fused_experts` seam survived even the
  self-healing reclassifier. Post-mortem: correct private capture needs
  capture-aware stream forking inside vLLM's runner — fork-grade. Router-
  split obsoletes the whole approach structurally: vLLM's own capture
  records the GEMM and the aux stream.

- **Eviction policy (LRU vs SLRU) — null under router-split.** Full matrix
  (slots {32, 48} × policy × fast-hit × concurrency) plus an N=3 ABAB
  repeat on the one suspicious column: dead ties, sign flips within the
  ±5% noise band; the earlier eager-world "+26% SLRU" claim also failed to
  reproduce on rerun. Post-mortem: under router-split there is no reachable
  policy-sensitive regime — policy moves *bytes* (SLRU streams ~2.5% fewer
  experts at slots=32), not decode *time*. SLRU stays default (never
  worse); tuning effort belongs elsewhere.

## 7. Future work

Plugin-scope:

1. **Formal transparency smoke**: sluice-piecewise vs vanilla-piecewise on
   a resident model, closing the one remaining inference in the piecewise
   correctness argument (§4.3).

Fork-grade / upstream (out of plugin scope by the analyses above):

2. **Cheaper piece dispatch in vLLM.** The c=1 structural floor is vLLM's
   piecewise piece-dispatch machinery itself: the noop-gap floor is 5.36 ms
   vs vanilla FULL's 3.7 ms — ~1.7 ms per step that no gap optimization can
   reclaim (staged-ids and lazy-step both proved the sync is no longer the
   wall). Also the specific c=1 bound on deep models: 48 × ~70 µs D2H on
   Qwen3.
3. **Quantized rsplit for marlin models.** Monolithic quant methods
   (V4-Pro's marlin MoE path) are skipped fail-closed today; a quantized
   traced entry would carry the win to the flagship offload target.
4. **Upstream PR of the restructure.** Land the router-split as a supported
   vLLM feature: gate + `select_experts` traced, a blessed streaming-gap
   op as a split point, and dispatcher support for size-branched traces —
   which would dissolve the envelope constraint (§3) rather than assert it.

---

*Flags for the shipped configuration:* `SLUICE_SLOTS=<K> SLUICE_PIECEWISE=1
SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1`, engine
`use_inductor_graph_partition=true`, `--max-num-batched-tokens <K//topk>`.
Raw data from the July 2026 campaign: `/work/results_pw/`,
`/work/results_rsplit/`, `/work/results_rsplit_qwen3/` on the
sluice-work PVC.
