<p align="center">
  <img src="assets/sluice-icon.png" alt="Sluice" width="120">
</p>

<h1 align="center">Sluice</h1>

<p align="center">
  <b>Routing-aware MoE expert offloading for vLLM</b> — a plugin, not a fork.
</p>

## What Sluice is

Sluice is a plugin for vLLM v0.23–v0.25 (a `vllm.general_plugins` entry point; setting
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

**DeepSeek-V2-Lite** (64 experts top-6, 1×H100, slots=48 → 16/64 offloaded),
decode tok/s. Corrected 2026-07-25 after a validity audit — see
"Correction: the c=8 figures" below.

| arm | c=1 | c=8 | output validity |
|---|---|---|---|
| classic eager-gap hook (previous plugin best) | 27 | 205 | bit-identical |
| **router-split + fast-hit, cudagraphs without Inductor** | **142.7** (7.01 ms) | 232.3 (34.4 ms) | **bit-identical** |
| vanilla vLLM, resident, FULL graphs | 265.9 (3.76 ms) | 1072.6 (7.46 ms) | reference |

Single-stream goes **27 → 143 tok/s (5.3×)**, reaching **54% of the vanilla
full-graph ceiling**, and the offloaded output is **bit-identical to stock** —
a stronger fidelity result than the scoped eager-only claim below. The
configuration that produces it keeps CUDA graphs but disables Inductor:

```bash
SLUICE_PIECEWISE=1 SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 \
VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
vllm serve deepseek-ai/DeepSeek-V2-Lite --max-num-batched-tokens 8 \
  --gpu-memory-utilization 0.50 --moe-backend triton \
  -cc.cudagraph_mode=PIECEWISE
```

At c=8 this configuration reaches 232.3 decode tok/s, **0.22× of resident
vanilla** — offloading does *not* cross the resident full-graph baseline at
this concurrency.

### Correction: the c=8 figures

Earlier releases of this table reported **1062.6** decode tok/s at c=8 for
"router-split + fast-hit + inductor partition", and claimed that at c=8 the
offloaded config *crosses* the vanilla full-graph baseline. **Both were
measured on a configuration that emits invalid output** and have been removed.

Router-split produces garbage tokens whenever it runs with Inductor
compilation **and** CUDA-graph capture together — the model emits degenerate
single-token runs rather than text. Measured facts:

- Either half alone is fine: Inductor without capture, and capture without
  Inductor (the config in the table above), are both bit-identical to stock.
- It is **not** a vLLM 0.25 regression — the same configuration emits the same
  garbage on **vLLM 0.23**, the version the original numbers were produced on.
  Those numbers were therefore never valid.
- It does not depend on slot count (36 / 48 / 60), on
  `max-num-batched-tokens` (1 / 6 / 8 / 10), on `use_inductor_graph_partition`,
  or on `SLUICE_RS_FAST_HIT` — all measured and excluded.

Root cause is not yet identified; the investigation, the run ledger and the
per-configuration validity verdicts are in
`.claude/sessions/results/PERF_LOG.md` (entries [7]–[15]) and
`runs.jsonl`. **Do not use the Inductor + capture combination until this is
resolved.**

**Qwen3-30B-A3B** (48 layers × 128 experts top-8, bf16, 1×H100 — small enough
that a true resident FULL-graph vanilla baseline exists).

> ⚠️ **UNVERIFIED — these router-split rows are at risk.** They were produced
> with Inductor partition **and** CUDA-graph capture, the same combination that
> is demonstrated above to emit invalid output on DeepSeek-V2-Lite (on both
> vLLM 0.23 and 0.25). The failure mechanism is a compiler/capture
> interaction, not a model-specific quirk, so these numbers are very likely
> affected the same way. **They have not been re-measured** — Qwen3-30B-A3B is
> not available in this environment's offline model cache — so they are
> flagged rather than corrected. Treat the router-split rows and the two
> comparative claims below as unverified pending a validity check.

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
pip install -e .   # into an environment that already has vLLM v0.23–v0.25
                   # (Sluice patches a small internal surface — pin vLLM)

# Plain offloading, eager: one env var
SLUICE_SLOTS=16 python examples/run_dsv4_ep4.py

# ROUTER-SPLIT: offloading + CUDA graphs.
# VLLM_USE_BREAKABLE_CUDAGRAPH=1 is REQUIRED: it keeps CUDA graphs but disables
# Inductor. Running router-split with Inductor AND capture together produces
# INVALID OUTPUT (see "Correction: the c=8 figures" above).
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
- **Compiled-mode numerics** — ⚠️ **RETRACTED 2026-07-25.** This previously
  read: "piecewise outputs differ from eager exactly the way vanilla vLLM's
  compiled mode differs from its eager mode (inductor fusions), no more."
  That is false. With Inductor **and** capture together, router-split output
  is *degenerate*, not merely numerically different. The controlled comparison:
  vanilla's own compiled-vs-eager delta stays coherent (5/6 greedy prompts
  still exactly match eager), while router-split's is 0/6 with degenerate
  token runs. See "Correction: the c=8 figures" above.
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
| `SLUICE_PIECEWISE=1` | off | piecewise CUDA graphs; streaming runs in an eager gap. Needs `cudagraph_mode=PIECEWISE` (FULL is refused — it would capture the gap) |
| `SLUICE_ROUTER_SPLIT=1` | off | the result above; requires `SLUICE_PIECEWISE=1` and the batch envelope (asserted) |
| `SLUICE_RS_FAST_HIT=1` | off | all-hit gaps skip LRU/map bookkeeping — exact (the D2H sync stays); +16% c=1, taken by 86–99% of gaps |
| `SLUICE_PROTECT_FRAC` | SLRU | eviction-policy knob; `0` = flat LRU. Measured to not move decode under router-split in any reachable regime — SLRU stays default (never worse, slightly fewer bytes streamed) |
| `SLUICE_GRAPH=1` | off | full CUDA graphs at **full residency** only (`static_full`); predates router-split |
| `SLUICE_HOOK_LITE=1` | off | classic-path gap-bookkeeping trim; single-digit % when steps are single-wave |

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
