# Design sketch — device-side gap kernel (H-KERNEL)

> # ☠️ DEAD — DO NOT BUILD THIS
>
> The gate failed. Entry [1] (`PERF_LOG_KERNEL.md`) measured the per-break cost
> at **2.27 µs**, not the ~452 µs this design assumed — **~200× smaller**.
> Adding 86 extra segment boundaries per step moved TPOT by **+0.20 ms**.
> Segment count is **0.5 %** of the 19.42 ms graph-shape term.
>
> Removing all 43 breaks — which is the entire point of the kernel below —
> would therefore recover **~0.1 ms of ~19 ms**. The premise is false.
>
> Kept as a record of the reasoning and of what the gate saved.

**Status: REFUTED.** Original framing preserved below.

---

## The causal chain this attacks

```
CPU must decide what to copy → needs topk_ids on the host
                             → needs a D2H sync
                             → needs cap.add_eager() (a segment boundary)
                             → 43 breaks/step  → 19.42 ms (55.5 % of V4's gap)
```

Every link is forced by the first. Tuning cannot reach it, which is why
`RS_STAGED` and `LAZY_STEP` both failed: they optimised the *work* and left the
*break* in place.

**Idea: move the decision onto the GPU.** Then there is no host round-trip, no
`add_eager`, and the whole model captures as ONE graph.

## Why this is capturable (the load-bearing claim)

CUDA graphs require a fixed **launch structure**, not fixed data. A single
kernel that internally branches on `topk_ids` is still *one launch node*. The
data-dependence lives inside the kernel, where a graph does not care.

So the gap becomes: `sluice_device_gap<<<...>>>(...)` — one captured node,
replayed every step, reading whatever the live buffers contain.

## What the kernel must do (all on-device)

| step | today (CPU, eager) | device version |
|---|---|---|
| read routing | D2H copy of `topk_ids` | already on device — read directly |
| hit test | Python dict `slot_of` | read `expert_map_buf[topk_ids]`, `-1` ⇒ miss |
| pick victims | Python SLRU (`protected`/`probation`) | device-resident age/clock array; see below |
| copy weights | `gpu[slot].copy_(cpu_store[local])` (DMA engine) | device loads from **mapped pinned host** memory |
| update map | `_write_map` + async H2D | direct store to `expert_map_buf` |

All state that Python owns today (`slot_of`, `expert_in_slot`, `free_slots`,
SLRU segments) must become **device-resident arrays**. That is the real work of
this change — the copy itself is the easy part.

## Eviction policy: simplify deliberately

Full SLRU (two segments, protected/probation, promotion on hit) is awkward in a
kernel. Proposed: **clock / second-chance** — one `age` byte per slot, cleared
on hit, incremented on sweep; evict the first slot with `age` above threshold.
Approximates LRU, is lock-free, and is a handful of instructions.

This is a **behaviour change**, so it must be justified empirically, not
assumed: the eviction-policy A/B already in the repo found SLRU vs flat LRU
"does not move decode under router-split in any reachable regime", which
suggests policy precision is not where the performance is. Re-measure anyway.

## The bandwidth risk (the main reason this could lose)

Device-side loads from mapped host memory go through the **load/store units**,
not the DMA copy engines. Expect meaningfully lower effective PCIe bandwidth
than `cudaMemcpyAsync` for large contiguous transfers.

Trade being made:

- **gain**: remove ~19.42 ms of segmentation (55.5 % of the gap)
- **risk**: inflate the PCIe term (currently 9.12 ms, 26 %) by some factor

If device-side copy is 2× slower per byte, the PCIe term goes 9.12 → ~18 ms and
the net gain collapses to ~10 ms. **This must be measured on a microbenchmark
before writing the full kernel** — a standalone "copy N MB host→device via
kernel loads vs via `copy_`" harness is a 30-line probe and kills the idea
cheaply if the ratio is bad.

## Staging plan (each stage kills the idea cheaply if it fails)

1. **[gate] entry [1]** — is the per-break slope ≥ 10 ms/K? *(running)*
2. **[gate] PCIe microbenchmark** — device-load vs `copy_` bandwidth ratio on
   this hardware. If worse than ~2×, stop.
3. **prototype, one layer** — device gap kernel on a single armed layer,
   correctness-gated against the CPU path (same slot placement sequence).
4. **all layers, no `add_eager`** — confirm `num_eager_breaks == 0` and that
   the model captures as one graph. Measure against the 29.25 ms floor.
5. **correctness** — the repo's bit-identical gate, then greedy-coherence on V4.

## Cheaper alternative that must be ruled out first

**Fewer breaks, not zero breaks.** One gap per *step* instead of per *layer*
takes 43 → 1 and recovers ~42/43 of whatever the segmentation term is, with no
CUDA at all. `LAZY_STEP` attempted this by speculation and lost 12–17 %, but it
was measured *before* entry [11] identified segmentation as the dominant term —
i.e. it was judged on the wrong scoreboard. **Re-running `LAZY_STEP` under
router-split on V4 is a flag-only experiment and should precede any kernel
work.**

If a Python change can capture most of the same prize, the kernel is not the
first move.
