<p align="center">
  <img src="assets/sluice-icon.png" alt="Sluice" width="120">
</p>

<h1 align="center">Sluice</h1>

<p align="center">
  <b>Routing-aware MoE expert offloading for vLLM</b> — a plugin, not a fork.
</p>

## What Sluice is

Sluice is a plugin for vLLM v0.23 (a `vllm.general_plugins` entry point; setting
`SLUICE_SLOTS` activates it, unset leaves vLLM untouched) that keeps a model's
MoE expert weights in host RAM and streams only the router-selected experts
into a small per-layer GPU slot cache each step. That lets models whose experts
exceed GPU memory — DeepSeek-V4-Pro's ~805 GiB checkpoint on a 320 GiB 4×H100
box, where stock vLLM and SGLang both OOM — serve inside vLLM's stack, and lets
models that *do* fit trade expert residency for KV headroom. The base offload
path, speculative-decode (MTP) results, and the engine comparison are in
[docs/EVALUATION.md](docs/EVALUATION.md) and [docs/COMPARISON.md](docs/COMPARISON.md);
this page leads with the newest result.

## ROUTER-SPLIT: offloading and CUDA graphs coexist

Expert streaming needs a per-step D2H sync and a mutable expert map — things a
CUDA graph cannot contain — so offloaded decode historically ran eager, far
below vLLM's graph-mode baseline. `SLUICE_ROUTER_SPLIT=1` restructures each MoE
layer at the runner's `_forward_entry` seam, without forking vLLM:

```
gate linear                 traced, captured
select_experts              traced, captured
vllm::sluice_stream_gap     the ONLY eager gap: ids D2H, stream missing
                            experts, refresh the expert map
vllm::fused_experts         traced, captured (the expert GEMM)
shared experts              traced, captured
```

The captured pieces bake **pointers** (slot buffers, expert-map buffer); the
gap rewrites their **contents** before each replay reads them. Everything else
runs under vLLM's own capture machinery.

> ⚠️ **Run router-split with `VLLM_USE_BREAKABLE_CUDAGRAPH=1`.** With Inductor
> compilation *and* CUDA-graph capture both active, router-split emits invalid
> output — see [#4](https://github.com/Etelis/sluice/issues/4). Rows marked ⚠️
> below were measured on that combination and are not valid.

**DeepSeek-V2-Lite** (64 experts top-6, 1×H100, slots=48 → 16/64 offloaded),
decode tok/s, measured in the July 2026 campaign:

| arm | c=1 | c=8 |
|---|---|---|
| classic eager-gap hook (previous plugin best) | 27 | 205 |
| router-split + fast-hit | 135.7 | 639.7 |
| router-split + fast-hit + inductor partition ⚠️ | 147.2 (6.79 ms) | 1062.6 (7.53 ms) |
| vanilla vLLM, resident, FULL graphs | 267.1 (3.7 ms) | 984.9 (8.1 ms) |

Single-stream goes **27 → 143 tok/s (5.3×)**, reaching 54% of the vanilla
full-graph ceiling, on the valid configuration. ⚠️ At c=8 the offloaded config
**crosses the vanilla
full-graph baseline**: 990–1066 tok/s across seven slots=48 runs vs 985 —
parity to +8% — while a quarter of the experts live in host RAM (replicated
on a heavier workload in the eviction-policy matrix).

**Qwen3-30B-A3B** (48 layers × 128 experts top-8, bf16, 1×H100 — small enough
that a true resident FULL-graph vanilla baseline exists):

| arm | c=1 | c=8 | c=12 |
|---|---|---|---|
| vanilla resident + FULL graphs | 210.1 (4.76 ms) | 742.3 (10.78 ms) | 700.7 (17.13 ms) |
| router-split, slots=64 (half offloaded) ⚠️ | 121.2 | **864.5** (9.25 ms) | — |
| router-split, slots=96 (quarter offloaded) ⚠️ | 122.8 | 883.7 | **1263.8** (9.5 ms) |

At c=8, offloading **half** the experts beats the resident baseline by **+16%**
(864.5 vs 742.3). Vanilla degrades past c=8 (TPOT 10.78 → 17.13 ms) while
router-split holds ~9.5 ms, so at matched c=12 the offloaded config leads by
**+80%** (1263.8 vs 700.7). Single-stream latency is **flat across the 25–62%
offload range** (122.8 / 121.2 / 119.2 tok/s, ~1% per tier) — under
router-split the offload *fraction* is nearly free at c=1, though c=1 itself
sits at ~58% of the resident baseline (the residual is vLLM's piece-dispatch
cost, measured by the no-op-gap floor, not streaming). On this model the
classic piecewise path trips a vLLM codegen assertion, so router-split is the
only offload+graphs path, not merely the fastest.

## Quickstart

```bash
pip install -e .   # into an environment that already has vLLM v0.23
                   # (Sluice patches a small internal surface — pin vLLM)

# Plain offloading, eager: one env var
SLUICE_SLOTS=16 python examples/run_dsv4_ep4.py

# ROUTER-SPLIT: offloading + CUDA graphs (worked example: Qwen3-30B, 80 GB H100)
# VLLM_USE_BREAKABLE_CUDAGRAPH=1 keeps CUDA graphs but disables Inductor; see #4.
SLUICE_SLOTS=96 SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve Qwen/Qwen3-30B-A3B \
  --max-num-batched-tokens 12 \
  --gpu-memory-utilization 0.40 \
  -cc.cudagraph_mode=PIECEWISE
```

Shorthand used below: `MAX_BT=<n>` stands for `--max-num-batched-tokens <n>`,
and `INDUCTOR_PARTITION=1` for `use_inductor_graph_partition: true` in the
compilation config.

Two sizing rules, both enforced or bitten in practice:

**1. The batch envelope: `max_num_batched_tokens <= slots // top_k`.**
Dynamo traces the model once, with the profile run's batch as the size hint,
and vLLM's dispatcher never re-evaluates guards — a Python size branch is
burned in at trace time. Bounding the envelope makes every step satisfy the
single-wave contract (tokens × top_k ≤ slots) by construction, so the trace
lands on the split path. Sluice **asserts this at config time** and the error
tells you the fix. Worked examples: V2-Lite top-6 at slots=48 → `MAX_BT=8`;
Qwen3 top-8 at slots=96 → `MAX_BT=12`. The envelope caps step size, so
aggregate throughput scales with slots, and prefill runs in `MAX_BT`-token
chunks — slow. Router-split is a **decode** product (a P/D decode worker, or
short-prompt serving).

**2. Budget slot memory *outside* `gpu_memory_utilization`.**
Slot buffers allocate **after** vLLM's memory profiling, so vLLM won't plan
around them — reserve their bytes by lowering `--gpu-memory-utilization`.
Worked example (Qwen3-30B on 80 GB): slots=96 per layer ≈ 43 GB of slot
buffers → `--gpu-memory-utilization 0.40` (vLLM plans weights+KV inside
32 GB, the slot cache takes its 43 GB afterwards). At slots=64 → 0.55, at
slots=48 → 0.45. Get this wrong and the slot allocation OOMs after profiling
succeeds.

## Correctness

- **Eager fidelity gate (V2-Lite): bit-identical to stock.** With graphs off,
  stock vs router-split at slots=48 produces md5-equal greedy token ids —
  26/26 layers armed, 8,000 gap calls. The traced gate + select_experts +
  fused_experts path reproduces stock kernels exactly.
- **Under forced streaming, still bit-identical**: at slots=36 (below the
  decode working set) the gate passes with **11,497 live misses** through the
  gap, fast-hit variant included; multi-token decode chunks are bit-identical
  under ~6,800 live misses. The miss path is fast as well as correct
  (125.5 tok/s c=1 at slots=36, streaming every ~1.5 gaps).
- **Red-team**: an adversarial pass worked through seven threat scenarios
  against the pointer/content split (capture passes, map staleness, envelope
  edges) and found zero bugs — see [docs/router-split.md](docs/router-split.md).
- **Compiled-mode numerics**: piecewise outputs differ from eager exactly the
  way vanilla vLLM's compiled mode differs from its eager mode (inductor
  fusions), no more.
- **Qwen3 ledger, stated plainly**: V2-Lite is bit-exact everywhere; on Qwen3
  the eager router-split output diverges from the classic hook. Forensics:
  each stack is bit-reproducible run-to-run, identical routing config
  consumed, all outputs coherent — deterministic fp differences between
  kernel stacks (modular apply vs monolithic `fused_experts`), the same
  legitimacy class as vLLM's own eager-vs-compiled drift. Not a logic bug.

## Flags

| flag | default | what it does |
|---|---|---|
| `SLUICE_SLOTS=N` | unset = plugin inert | N expert slots per layer per rank; experts stream from host RAM |
| `SLUICE_PIECEWISE=1` | off | CUDA graphs on; streaming runs in an eager gap |
| `SLUICE_ROUTER_SPLIT=1` | off | the result above; requires `SLUICE_PIECEWISE=1` and the batch envelope (asserted) |
| `SLUICE_RS_FAST_HIT=1` | off | all-hit gaps skip LRU/map bookkeeping — exact (the D2H sync stays); +16% c=1, taken by 86–99% of gaps |
| `SLUICE_PROTECT_FRAC` | SLRU | eviction-policy knob; `0` = flat LRU. Measured to not move decode under router-split in any reachable regime — SLRU stays default (never worse, slightly fewer bytes streamed) |
| `SLUICE_GRAPH=1` | off | full CUDA graphs at **full residency** only (`static_full`); predates router-split |
| `SLUICE_HOOK_LITE=1` | off | classic-path gap-bookkeeping trim; single-digit % when steps are single-wave |

### Set for you — you should not need to pass these

| flag | default | why |
|---|---|---|
| `VLLM_USE_BREAKABLE_CUDAGRAPH=1` | **pinned by the plugin** when `SLUICE_ROUTER_SPLIT=1` | **Correctness, not tuning.** Router-split emits *invalid output* when Inductor compilation and CUDA-graph capture are both active ([#4](https://github.com/Etelis/sluice/issues/4)). This makes vLLM skip torch.compile (`mode=NONE`) while keeping capture, so the eager gap comes from `add_eager()` rather than the fx splitter. vLLM auto-enables it for DeepSeek-V4/MiniMax only — every other MoE model had to be told, and forgetting produced plausible-but-wrong tokens with no error. Pass `=0` to opt out (only sane with router-split off). |
| `SLUICE_ALLOW_FULL_CG=1` | **on** | Permits vLLM's default `cudagraph_mode=FULL_AND_PIECEWISE` on the breakable path, where `add_eager()` cuts the capture whatever the mode says — so a "full" graph cannot swallow the D2H sync. Forcing plain `PIECEWISE` throws away vLLM's full **decode** graphs: measured **+16 ms/step on V4 with no Sluice in the process at all**. Still refused on the fx-splitting path, where a full graph genuinely would capture the hook. |

Consequences worth knowing:

* **Do not pass `-cc.cudagraph_mode=PIECEWISE`.** It is what the second flag
  exists to avoid. On V4 at TP=4: 41.7–44.5 ms forced-PIECEWISE vs **36.4 ms**
  letting vLLM keep its decode graphs.
* **Do not hand-tune `--gpu-memory-utilization` down for Sluice.** That was a
  workaround for a bug — vLLM's memory profiler could not see the slot cache,
  so it sized the KV cache as if the cache did not exist and OOMed. Fixed;
  use the same value as vanilla, or none at all.
* **`--max-num-batched-tokens <= SLUICE_SLOTS // top_k` is still required**
  under router-split and cannot be relaxed. Router-split is single-wave by
  contract; the oversized-step fallback is the classic hook, which refuses to
  run under capture. Relaxing it was tried — the server starts and then dies
  mid-run.

**Negative results, kept in-tree, OFF, with warnings** — they are correct but
slower, and the post-mortems say why:

| flag | verdict |
|---|---|
| `SLUICE_RS_STAGED=1` | doorbell-staged ids instead of the gap's stream sync — bit-identical, engaged, **−6–8% on both models**: under inductor partition the sync it replaces is already nearly free |
| `SLUICE_LAZY_STEP=1` | one-sync-per-step optimistic forward with exact rerun-on-miss — bit-identical, **−12–17%**: piecewise had already demoted the per-layer syncs it collapses |
| `SLUICE_GEMM_GRAPH=1` | private CUDA-graph capture of the expert GEMM — **parked** after five documented capture hazards (the fifth, aux-stream shared experts, needs vLLM-side work); router-split supersedes it |

**Diagnostic:** `SLUICE_RS_NOOP=1` replaces the gap with a no-op to measure the
structural floor — **outputs are invalid**, never serve with it.

## Where the details live

- [docs/router-split.md](docs/router-split.md) — design, the pointer/content
  contract, envelope rationale, red-team log, the measurement record (every
  arm above, the failure modes, the levers that didn't pay), the eager-hook
  prehistory, and the eviction-policy A/B — including the retraction of an
  earlier +26% claim that failed to reproduce
- [docs/EVALUATION.md](docs/EVALUATION.md) — the base offload campaign:
  V4-Pro on 4×H100 where stock OOMs, MTP speculative decode (+67% c=1),
  working-set sizing, limits
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — load/forward paths, gotchas
- [docs/COMPARISON.md](docs/COMPARISON.md) — vs stock vLLM, SGLang,
  KTransformers, llama.cpp

---

<p align="center">
  Apache-2.0 · built on <a href="https://github.com/vllm-project/vllm">vLLM</a> ·
  <a href="docs/ARCHITECTURE.md">Architecture</a> ·
  <a href="docs/COMPARISON.md">Comparison</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>
