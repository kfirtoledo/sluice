# H-KERNEL log — device-side gap kernel investigation

Worktree `perf/device-gap-kernel`, pod `sluice-kx`. Separate from the
`perf/h1-router-split-v4` agent (pod `sluice-h9`).

---

### [1] Is the 19.42 ms "graph-shape" term driven by the NUMBER of eager breaks?
**2026-07-26 · REFUTED · the kernel is NOT justified**

- **Hypothesis**: entry [11]'s graph-shape term (19.42 ms, 55.5 % of V4's gap
  to vanilla) scales with the count of breakable-cudagraph segment boundaries
  (43 `add_eager`/step). If so, a device-side gap removing all breaks recovers
  most of it.
- **Prediction** (pre-registered in `PREDICTION.md`, commit e45e92d): H-linear
  ⇒ +19.4 ms per K; H-small ⇒ +0.9–2.2 ms per K. Decision rule fixed in
  advance: **≥10 ms/K justifies the kernel, ≤3 ms/K kills it.** Stated prior:
  60/40 toward H-small.
- **Instrument**: `SLUICE_EXTRA_BREAKS=K` — K extra **empty**
  `cap.add_eager(_sluice_noop_break)` per gap. Adds boundaries while holding
  sync, PCIe, map writes and the clone fixed.
- **Setup**: V4-Flash, TP=4/EP, marlin, fp8 KV, slots=48, `MAX_BT=8`,
  `-cc.cudagraph_mode=PIECEWISE`, `SLUICE_RS_NOOP=1`, random 32/200 c=8,
  np=64. Warm + 2 measured repeats per arm. Pod `sluice-kx`, node
  pokprod-b93r43s2.

**Measured** (TPOT ms, r1/r2, warm discarded):

| K | extra breaks/step | r1 | r2 | mean | spread |
|---|---|---|---|---|---|
| 0 | 0 | 28.81 | 28.38 | **28.59** | 1.5 % |
| 1 | +43 | 34.63 | 34.28 | **34.45** | 1.0 % |
| 2 | +86 | 28.21 | 29.37 | **28.79** | 4.0 % |

- **K=0 reproduces entry [11]'s 29.25 ms no-op floor** (28.59 here, −2.3 %,
  different node) — the arms are comparable and the instrument is sane.
- **K=2 ≈ K=0.** Adding **86** extra segment boundaries per step changed TPOT
  by **+0.20 ms**.

**Slope (K=0 → K=2): 0.098 ms per K = 2.27 µs per break.**
Predicted under H-linear: 451.7 µs per break. **Measured is ~200× smaller.**

⇒ Segment count accounts for **0.098 ms of the 19.42 ms term — 0.5 %.**

- **Verdict: DEAD.** Slope is 0.098 ms/K against a 3 ms/K kill threshold. The
  premise of the device-side kernel is false: eliminating all 43 breaks would
  recover ~0.1 ms, not ~19 ms. **`DESIGN-device-gap.md` is not to be built.**
  My pre-registered prior (60/40 toward H-small) was correct, but the true
  effect is far smaller than even H-small predicted (2.27 vs 20–50 µs/break).

**Unexplained anomaly, recorded not dismissed.** K=1 sits 20 % above both K=0
and K=2, and its two repeats agree to 1 % — so it is a stable property of that
server instance, not run noise. It is inconsistent with break-count causation
in either direction (K=2 would have to be higher still). Most likely a
per-instance placement/clock/co-tenant effect. It does not rescue the
hypothesis: taking K=1 at face value gives +5.9 ms/K, still below the 10 ms/K
bar, and K=2 contradicts it outright. **A third repeat at K=1 on a fresh server
would be the way to settle it** — not done here, since no reading of it
revives the kernel.

**Where the 19.42 ms actually is** — by elimination, NOT in the segment
boundaries. Remaining candidates inside the per-gap 451.7 µs, in the order I
would probe them:

1. **`hidden_states.clone()`** (`offloader.py`, end of `_sluice_stream_gap`) —
   43 clones/step. Present in the real gap, absent from my empty breaks.
2. **torch custom-op dispatch** for `vllm::sluice_stream_gap` — dispatcher,
   fake-tensor and mutation bookkeeping, 43×/step. Also absent from my empty
   breaks (a bare Python callable).
3. **Structural cost of being segmented at all** — a fixed penalty independent
   of *how many* segments (e.g. lost cross-segment overlap, or the traced entry
   replacing the fused `vllm::moe_forward` with separate gate/select/GEMM ops).
   This one would NOT scale with K, and is fully consistent with what I
   measured.
4. **Per-rank convoy** across the 4 ranks.

Note that (3) is the reading most consistent with the data, and it is the one
*least* addressable by a kernel — it is a property of the router-split
restructuring itself, not of the gap.

**Next probes (all cheap, flag- or one-line-sized; none require CUDA):**

- **[a] kill the clone.** Return a mutated-in-place tensor or restructure the
  op so no clone is needed; measure. Directly tests candidate 1.
- **[b] gap-op dispatch cost.** Register a second custom op that only does
  `return hidden_states.clone()` (no `add_eager`, no gap body) and add it to
  the traced entry; the delta isolates dispatch + clone from the break.
- **[c] re-run `LAZY_STEP` under router-split on V4.** Takes 43 gaps → 1. It
  lost 12–17 % when judged before segmentation was understood; on the current
  scoreboard it targets candidates 1+2 at 43× leverage, flag-only.
- **[d] fixed-vs-scaling test**: arm router-split on only *half* the layers.
  If the term is structural (candidate 3) it should fall roughly in half; if it
  is per-gap work it should also halve — this separates neither, so run [a]/[b]
  first.

**Artifacts**: `PREDICTION.md` (pre-registration), `DESIGN-device-gap.md` (now
shelved with a DEAD banner), instrument in `offloader.py`
(`SLUICE_EXTRA_BREAKS`, diagnostic-only, safe to keep — it is inert at K=0).
Raw: `/work/breaks.txt` in pod `sluice-kx`.

---

### [2] The DOWNWARD direction: does removing breaks help? **No.**
**2026-07-26 · "fewer cuts" is dead as a strategy**

Entry [1] only *added* breaks (43→86→129). A flat response to adding does not
logically prove a flat response to removing — the curve could have a knee
below 43. This closes that gap.

- **Instrument**: `SLUICE_BREAK_LAYERS=N` — only the first N layers call
  `add_eager`; the rest run the (no-op) gap body inline. Under `SLUICE_RS_NOOP=1`
  the suppressed call would have done nothing, so this changes break count and
  nothing else. Same config/pod/node as entry [1] (`sluice-kx`, fresh instance
  after an eviction).

**Measured** (TPOT ms, r1/r2, warm discarded):

| N (layers that break) | breaks/step | r1 | r2 | mean | spread |
|---|---|---|---|---|---|
| 43 (normal) | 43 | 30.28 | 30.54 | **30.41** | 0.9 % |
| 1 | 1 | 28.64 | 28.72 | **28.68** | 0.3 % |
| 0 | 0 | 33.83 | 33.87 | **33.85** | 0.1 % |

- **43 → 1 break: −1.73 ms.** Removing **42** breaks recovers 1.73 ms of the
  19.42 ms term = **8.9 %**. Per break: **41 µs**.
- **1 → 0 breaks: +5.17 ms — removing the LAST break makes it WORSE.**
- **Best case (1 break) is 28.68 ms against vanilla's 9.83 ms — still 2.9×.**

**Verdict: the "fewer cuts" strategy is DEAD, in both directions.**
Driving breaks to their practical minimum buys ~9 % of the term and leaves V4
at 2.9× vanilla. This also removes the rationale for re-running `LAZY_STEP`
against the floor (candidate C3 in the `sluice-cut-cost` agent): its whole
prize was 43 gaps → 1, which is now measured at 1.73 ms — far less than the
12–17 % it previously cost.

**Consistency with entry [1], and an honest discrepancy.** Both directions
agree that breaks are *cheap* relative to the 452 µs the naive model assumed,
but they disagree on magnitude: **2.27 µs/break** (additive) vs **41 µs/break**
(subtractive) — 18×. Neither is near 452. Candidate explanations: added breaks
land at a different point in the segment than removed ones; or the subtractive
delta is contaminated by the inline-body path. Not resolved. The conclusion is
robust to either value (43 × 41 µs = 1.76 ms ≪ 19.42 ms), but the per-break
number itself should be quoted as "tens of µs, direction-dependent", not as a
precise constant.

**Unexplained: why is 0 breaks WORSE than 1?** +5.17 ms, with 0.1 % spread on
both repeats, so it is not noise. Hypotheses, none tested: a zero-break capture
takes a different code path in vLLM's `BreakableCUDAGraphCapture`; a single
undivided graph has worse memory or launch behaviour; or the fallback that runs
the gap body inline during capture perturbs something. **This is the one
genuinely surprising number in the campaign and deserves a probe** — if a
single undivided graph is *structurally* worse, that has implications well
beyond Sluice. Flagged, not explained.

**Net implication for the 19.42 ms**: neither the count nor the presence of
breaks explains it. Suspicion moves further onto per-gap *work* (the clone,
op dispatch) and onto the *restructuring* — candidates C1, C2, C4 in the
`sluice-cut-cost` agent. C3 is closed.

---

### [3] Is the 24 ms at zero breaks the `hidden_states.clone()`?
**2026-07-26 · INCONCLUSIVE — and it invalidated the instrument**

- **Hypothesis**: C1. With N=0 breaks the graph has vanilla's shape, yet
  router-split was 24.02 ms slower (entry [2]); the gap op's tail
  `hidden_states.clone()` runs 43×/step and was the leading suspect.
- **Instrument**: `SLUICE_NO_CLONE` (0=clone, 1=passthrough, 2=`empty_like`).
  At N=0 + mode 1 the gap op is a **pure passthrough**, so the only thing left
  distinguishing the arm from vanilla is router-split's op structure.
- **Pre-registered** in `PREDICTION-2.md` (commit 189de94), with the
  arithmetic the hypothesis had to beat: `[8, 7168]` bf16 ≈ 114 KiB per clone,
  43/step ≈ 4.8 MiB, ~3 µs of HBM bandwidth, **≤ ~0.5 ms even charging 10 µs
  of launch overhead each**. Stated prior: **85/15 that the clone is NOT the
  cause.** Decision rule: drop > 3 ms ⇒ promote C1; drop < 1 ms ⇒ C1 dead.

**Measured** (TPOT ms, `bench serve`, same server instance per arm):

| arm | warm | r1 | r2 | mean(r1,r2) |
|---|---|---|---|---|
| N=0, clone | 28.90 | 28.63 | 28.88 | **28.76** |
| N=0, no clone | 27.29 | 32.54 | 33.35 | **32.95** |

**Removing a memory copy made it 4.2 ms SLOWER.** That is not a physical
result. The no-clone arm also degraded **monotonically within its own three
runs** (27.29 → 32.54 → 33.35) while the clone arm held flat to 0.9 %.

⇒ **What this actually measured is the instrument.** There is **~6 ms (22 %)
of drift within a single server instance**, and it is larger than the effect
under test. Not thermal or co-tenant: clocks pinned at 1980 MHz, no throttle
bits, temps 26–32 °C, and the other agent's pod is on a different node
(`sluice-g25` on b93r43s2, mine on b93r39s1). Most likely accumulation inside
the long-lived server across successive `bench serve` runs (prefix cache / KV
block state / scheduler), which the monotonic shape fits.

**Verdict on C1: INCONCLUSIVE, bounded.** The clone's effect is somewhere in
|Δ| < ~5 ms. That is *consistent* with the ~0.5 ms the arithmetic predicts but
does not confirm it. **C1 is not promoted and not closed.** Two independent
arguments still say it is small, and neither depends on this run: the
bandwidth arithmetic above, and the fact that **at N=0 a captured graph
replays with no Python at all** — so per-step dispatch, fake-tensor
bookkeeping and Python call overhead are paid once at capture, not per step.
Anything living only in Python cannot be charged 24 ms/step.

**Retroactive correction to entries [1] and [2].** Their *large* conclusions
survive, because they are 20×-scale effects: breaks cost tens of µs, not
452 µs; the cut is not the cost. Their *fine structure* does not survive this
noise floor and should no longer be quoted as measured:

- the **1.73 ms** from removing 42 breaks (entry [2]) — inside the band;
- the **5.17 ms** "0 breaks is worse than 1 break" anomaly (entry [2]) — inside
  the band, and directly contradicted here: the same N=0 config measures
  **28.76 ms** on this node against entry [2]'s **33.85 ms**. That anomaly was
  most likely instance drift, not a property of zero-break capture. It is
  withdrawn as a finding.
- the **2.27 vs 41 µs/break** discrepancy between the two directions — both
  values are noise-dominated; only "tens of µs, ≪ 452" is supportable.

**Instrument change for entry [4]**: `vllm bench latency` instead of
`bench serve`. In-process (no API server, no network, no cross-run
accumulation), 10 warmup + 30 measured iterations, so each arm yields a
distribution rather than two samples. The vanilla anchor is measured twice,
first and last, so drift is measured rather than assumed.

---

### [4] Does the cost scale with armed layers? **There is no cost to scale.**
**2026-07-26 · THE 19.42 ms "GRAPH-SHAPE" TERM DOES NOT EXIST AT MATCHED FLAGS**

- **Hypothesis**: C4. If the 24 ms at zero breaks is router-split's op
  restructuring (5 ops/layer replacing vanilla's fused `vllm::moe_forward`),
  TPOT should climb roughly linearly as 0 → 11 → 22 → 43 layers are armed.
- **Instrument**: `SLUICE_RS_LAYERS=N` (arm only the first N MoE layers; under
  `SLUICE_RS_NOOP=1` the classic streaming hook is a passthrough too, so
  nothing streams in any arm and N is the only variable). Measured with
  **`vllm bench latency`** — in-process, no API server, 10 warmup + 30
  measured iterations per arm — after entry [3] showed `bench serve` drifting
  ~6 ms within an instance. **A vanilla arm ran first AND last.**
- **Pre-registered** in `PREDICTION-2.md`: linear ⇒ C4 confirmed; flat-then-step
  ⇒ fixed capture-mode effect; N=0 already ≫ vanilla ⇒ the cost is Sluice being
  attached at all. Stated prior: **70/30 toward linear.**

**Measured** (batch 8, in 8 / out 200, 30 iterations; per-step = avg/200):

| arm | armed layers | avg latency | per-step | Δ vs vanilla mean |
|---|---|---|---|---|
| vanilla_a | none (no Sluice) | 5.899 s | 29.50 ms | −0.73 |
| rs0 | 0 | 6.313 s | 31.56 ms | +1.34 |
| rs11 | 11 | 5.568 s | 27.84 ms | **−2.38** |
| rs22 | 22 | 5.968 s | 29.84 ms | −0.39 |
| rs99 | **43 (all)** | 5.988 s | 29.94 ms | **−0.28** |
| vanilla_b | none (no Sluice) | 6.191 s | 30.96 ms | +0.73 |

`ARMED:` was read back from each log — 0, 11, 22, 43 as intended, and `none`
in both vanilla arms.

**The two vanilla anchors differ by 0.292 s = ±1.5 ms/step**, for byte-identical
configs. All six arms span 0.745 s (13 %). Arming *all 43* layers lands
**between** the two vanilla runs.

**Verdict: C4 REFUTED, and with it the premise of the whole investigation.**
Full router-split costs **< 1.5 ms/step** against vanilla at matched flags —
i.e. below the instance noise floor. My pre-registered prior (70/30 linear) was
wrong, and so was the "flat-then-step" alternative: the response is **flat at
zero**. There is no per-layer restructuring cost, no fixed capture-mode cost,
and nothing for a kernel, a re-fusion, or a cut-count change to recover.

#### Where the 19.42 ms actually came from: an unmatched baseline

The ledger's entry [11] computes the term as `noop floor (29.25) − vanilla
(9.83)`. Both numbers are real. **The subtraction is not**, because the two
arms did not share flags. From the ledger's own setup block:

> **The one thing that differs** is the router-split bundle: … and
> `--max-num-batched-tokens 8` … These are not separable — they are what
> "enable router-split" *means* on this model.

`--max-num-batched-tokens 8` is router-split's smallness envelope
(`slots // top_k` = 48/6 = 8). It was present in **every** Sluice arm and in
**no** vanilla arm. It was flagged at the time as a **TTFT** risk (prefill
chunking) and was never suspected of moving TPOT, so its whole effect was
booked as "breakable-cudagraph segmenting".

Measured here: vanilla **with** that flag decodes at **29.5–31.0 ms/step**,
against the ledger's 9.83 ms for vanilla without it. That single flag accounts
for essentially the entire "graph-shape" term. Entry [5] (the 2×2:
{marlin, default backend} × {MAX_BT 8, default}, no Sluice in any arm)
attributes it precisely. Note marlin cannot be the asymmetry — the ledger
confirms it was in both arms.

#### What this retracts

- **The 19.42 ms graph-shape term (55.5 % of V4's gap)** — withdrawn. It is a
  baseline artifact, not a property of router-split.
- **"Even a perfect offloader leaves V4 at 3× vanilla, purely from how the
  graph is segmented"** (entry [11]'s central conclusion) — withdrawn. The
  no-op floor is ~vanilla once vanilla carries the same envelope flag.
- **"breakable-graph segmentation: 19.42 ms — redesign; not reachable by any
  flag"** — withdrawn. It was reachable by a flag: the one on the *other* arm.
- Entries [1] and [2] of this log measured real things (breaks are cheap) but
  were **answering a question that should not have been asked**.

#### What survives

The gap that is actually Sluice's to close is `rs48 − noop` = **15.58 ms**:
D2H sync of `topk_ids`, 4-rank convoy, map write, PCIe H2D. That term was
measured *within* matched arms and is unaffected by this correction. It is
also the term with a real engineering story (slots 48 → 62 recovers 9.12 ms of
it by tuning alone).

**And the envelope itself is now the headline cost.** If `MAX_BT=8` costs
~20 ms/step on its own, then router-split's requirement to cap the batch at
`slots // top_k` tokens is *the* dominant penalty of the design — far larger
than anything in the gap. Raising that cap means raising `slots` or lowering
`top_k` per wave, which is a **tuning and capacity** question, not a kernel
one. That is where the next work belongs.

---

### [6] THE DECISIVE ARM — redo the ledger's subtraction with a matched vanilla
**2026-07-26 · CONFIRMED: the 19.42 ms is two flags, not the offloader**

Entries [4] and [5] used `vllm bench latency`; the ledger used `bench serve`.
A skeptic should ask whether the **instrument**, not the flags, produced the
difference. This entry was written to be able to **retract entry [4]**, and it
did not.

- **Design**: three arms, `bench serve`, one node, one session, warm + 3
  measured repeats each. The ledger's router-split bundle contains **three**
  things its vanilla arm lacked — the `SLUICE_*` env, `--max-num-batched-tokens
  8`, and `-cc.cudagraph_mode=PIECEWISE`. Arm B adds the last two to vanilla
  and **nothing else**. That arm had never been run.

| arm | flags vs vanilla | Sluice | TPOT |
|---|---|---|---|
| **A** vanilla, ledger flags | — | no | **9.83 ms** (9.87/9.83/9.84/9.83) |
| **B** vanilla + MAX_BT=8 + PIECEWISE | +2 | **no** | **29.1 ms** (29.13/29.01/29.21) |
| **C** router-split, `RS_NOOP` | +2 +env | yes | ledger: **29.25 ms** |

- **Arm A reproduces the ledger to three digits**: 787.0 tok/s here against
  786.6 recorded, TPOT 9.83 both. The node, build and workload are faithful, so
  arm B is a real comparison and not a setup difference.
- **Arm B lands on the no-op floor**: 29.1 ms against the ledger's 29.25 ms —
  **0.5 % apart** — with **no offloader in the process at all**.

⇒ **Two flags cost ~19.3 ms/step by themselves.** That is the entire
"graph-shape" term, reproduced without Sluice.

**Verdict: the retraction stands, now in the ledger's own instrument.** The
19.42 ms was `29.25 − 9.83` across arms differing by two flags. Both numbers
were real measurements; the subtraction was not. Entry [4] is confirmed rather
than overturned, and `bench latency` was not the cause of its result.

**Correction to entry [3].** I attributed ~6 ms of drift to `bench serve` as an
instrument. That was wrong: arm A is stable to **0.4 %** across four runs, and
arm B at gpu-util 0.55 to **0.7 %**. The instability belongs to **the MAX_BT=8
configuration under some conditions** (arm B at 0.75 swung 29.8–33.1, ~10 %),
not to the harness. Entry [3]'s clone result is still inconclusive — it ran in
the unstable regime — but the diagnosis of *why* was mistaken.

**Operational notes.** Arm C OOMs at `--gpu-memory-utilization 0.75` (needs
53.87 GiB for KV on top of Sluice's slot buffers); the Sluice arms must run at
0.55, which is what the ledger did. Arm A reproducing 9.83 at 0.75 shows
gpu-util is not what moves these numbers. Killing a run leaves orphaned
`VLLM::Worker_TP` processes holding ~62 GiB/GPU — check
`nvidia-smi --query-compute-apps` and `kill -9`.

#### What is now the actual cost of "enabling router-split"

| term | ms | status |
|---|---|---|
| **smallness envelope** (MAX_BT ≤ slots//top_k, + PIECEWISE) | **~19.3** | **REAL, and the dominant term. Not yet split between the two flags.** |
| sync + convoy + map + PCIe (`rs48 − noop`) | 15.58 | real, measured within matched arms |
| router-split's op restructuring | **< 1.5** | entry [4]: below the noise floor |
| segment count (43 breaks) | ~0.1 | entries [1]/[2] |

The next question is which of the two flags carries the ~19.3 ms, and why a
capture mode or a batch cap should cost that much on a model whose compilation
mode is already `NONE`. `scripts/cgmode.sh` crosses them, no Sluice in any arm.

#### [6b] Triangle closed — B and C as a matched pair at the ledger's gpu-util

Arm C OOMs at 0.75, so B and C were re-run together at **0.55** (the ledger's
value). Arm A reproducing 9.83 ms at 0.75 already showed gpu-util is not what
moves these numbers.

| arm | flags vs vanilla | Sluice | warm | r1 | r2 | r3 | **mean(r1-r3)** |
|---|---|---|---|---|---|---|---|
| **A** vanilla, ledger flags | — | no | 9.87 | 9.83 | 9.84 | 9.83 | **9.83** |
| **B** vanilla + MAX_BT=8 + PIECEWISE | +2 | **no** | 29.13 | 29.01 | 29.21 | 29.10 | **29.11** |
| **C** router-split `RS_NOOP` | +2 +env | yes (**43 layers armed**) | 31.13 | 29.45 | 29.02 | 29.03 | **29.17** |

**Both endpoints of the ledger's subtraction reproduce faithfully:**
A = 9.83 vs ledger 9.83; **C = 29.17 vs ledger's no-op floor 29.25 (0.3 %)**.
`ARMED: router-split layers=43` confirms C is the real router-split config, and
`ARMED:` is empty in A and B.

**The decomposition:**

| step | Δ | what it is |
|---|---|---|
| A → B | **+19.28 ms** | **two flags. No offloader in the process.** |
| B → C | **+0.06 ms** | **ALL of router-split**: 43 breaks, 43 clones, 5 ops/layer × 43 layers |

⇒ Router-split's structural cost is **0.06 ms/step**, an order of magnitude
below the noise floor and ~320× smaller than the 19.42 ms attributed to it.
The ledger's two anchor measurements were both correct; the arm between them
was missing, and its absence turned a flag cost into a design verdict.

**This is the campaign's final answer.** The "cut cost" does not exist in any
form: not segment count (~0.1 ms), not the clone (< 5 ms bound, ~0.5 ms by
arithmetic), not the op restructuring (0.06 ms measured here, < 1.5 ms in
entry [4]). Everything that made V4 look structurally capped was
`--max-num-batched-tokens 8` and `-cc.cudagraph_mode=PIECEWISE`.

---

### [7] WHICH envelope flag? The 2×2. **It is `cudagraph_mode`, not the batch cap.**
**2026-07-27 · pod `sluice-kx` rebuilt on node pokprod-b93r39s2**

Entry [6] proved the pair costs +19.28 ms with no Sluice in the process, but
`--max-num-batched-tokens 8` and `-cc.cudagraph_mode=PIECEWISE` had only ever
moved together. Full 2×2, `bench serve`, one script, one gpu-util (0.75),
**no Sluice in any arm**, warm + 3 measured repeats:

| arm | flags | r1 | r2 | r3 | **mean** | Δ vs A |
|---|---|---|---|---|---|---|
| **A** | neither | 9.74 | 9.79 | 9.90 | **9.81** | — |
| **D** | MAX_BT=8 only | 17.15 | 17.29 | 17.27 | **17.24** | +7.43 |
| **E** | PIECEWISE only | 25.38 | 25.86 | 26.31 | **25.85** | **+16.04** |
| **B** | both | 29.46 | 29.56 | 29.10 | **29.37** | +19.56 |

Arm A reproduces 9.83 from the other node (0.2 %), and arm B reproduces 29.11
(0.9 %) — the 2×2 is anchored to entry [6] on both corners.

**Marginal costs (what a fix would actually buy):**

| move | Δ | |
|---|---|---|
| add PIECEWISE **given** MAX_BT=8 | **+12.13 ms** | ← the recoverable prize |
| add MAX_BT=8 **given** PIECEWISE | +3.52 ms | |

Sub-additive: 7.43 + 16.04 = 23.47 against a joint 19.56, so the two overlap by
~3.9 ms.

**Why.** V4's **default** `cudagraph_mode` is **`FULL_AND_PIECEWISE`** (read
from arm A's config dump), under which vLLM captures **full graphs for
pure-decode batches** and uses piecewise only for mixed prefill-decode.
Forcing plain `PIECEWISE` discards the full decode graphs — and a decode
benchmark is entirely that regime. This is a vLLM configuration effect with no
offloader involved.

**The batch cap is NOT negotiable.** `_check_config` sets `traced = True`
whenever capture is on, and then enforces `max_num_batched_tokens <=
slots // top_k`. Under EP with TP=4 each rank holds 256/4 = **64** experts per
layer, so slots ≤ 64 and the envelope caps at **10 tokens**. "Raise slots to
widen the envelope" — which I proposed before checking — is not available in
this configuration. Its whole prize would have been 3.52 ms anyway.

⇒ **The target is the ~12 ms from keeping vLLM's full decode graphs.**

Sluice refuses them: `if self.piecewise and "FULL" in cg_name: raise`, on the
grounds that FULL would capture the hook's D2H sync. That is **correct for the
fx-splitting path** (V2-Lite, Qwen3), where the gap exists only because
`moe_forward` is in `splitting_ops`. It is **over-broad for V4's breakable
path**, where compilation mode is `NONE`, `splitting_ops` is empty, and the
gap comes from `BreakableCUDAGraphCapture.add_eager()` — which ends and
reopens a segment whatever `cudagraph_mode` says. Entry [2] already showed that
machinery yielding one undivided graph.

`SLUICE_ALLOW_FULL_CG=1` permits it **only** when the breakable path is
actually active (`mode == NONE` and no `splitting_ops`), leaving the refusal
intact everywhere it is right. Entry [8] gates it on correctness first: a
frozen expert map produces plausible-but-wrong tokens, not an error.

---

### [8] FIRST MEASURED WIN — keep vLLM's FULL decode graphs: **−28 % TPOT**
**2026-07-27 · `SLUICE_ALLOW_FULL_CG=1`, V4-Flash, TP=4/EP, slots=48, real streaming**

Entry [7] showed forcing `PIECEWISE` costs +12.13 ms marginal with no Sluice in
the process. This tests whether router-split can simply stop forcing it.

**Mechanism first — confirmed, not assumed:**

```
Sluice: permitting cudagraph_mode=FULL_AND_PIECEWISE because the BREAKABLE
        capture path is active (compilation mode NONE, no splitting_ops)
ROUTER-SPLIT armed on 43/43 MoE layers
router-split layers=43 gap-calls=54000 misses=31677 fast-hits=32894
Capturing CUDA graphs (decode, FULL)
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)
```

vLLM captures **both** families; the gap **still fires every step** under FULL
(54 000 calls, 31 677 real misses). A swallowed D2H sync would have flatlined
those counters and frozen the map. It did not.

**Measured** (`bench serve`, random 32/200, c=8, np=64, warm + 3 repeats, same
node and session, identical Sluice settings — `cudagraph_mode` is the ONLY
variable):

| config | r1 | r2 | r3 | **mean** | spread | tok/s |
|---|---|---|---|---|---|---|
| forced `PIECEWISE` (shipping) | 46.94 | 51.28 | 50.33 | **49.52** | 9 % | 132.5 |
| **`FULL_AND_PIECEWISE` (new)** | 35.73 | 35.52 | 35.74 | **35.66** | **0.6 %** | **182.4** |

⇒ **−13.86 ms/step (−28 %), +37.7 % output throughput.** Against the control's
*best* repeat it is still −11.2 ms. The new config is also **15× more stable**
(0.6 % vs 9 % spread), which matches entry [6]'s finding that the instability
belongs to the forced-PIECEWISE regime.

**Decomposition against each config's OWN matched vanilla** (entry [7] arms):

| | Sluice | matched vanilla | streaming overhead |
|---|---|---|---|
| old (forced PIECEWISE) | 49.52 | 29.37 (arm B) | 20.15 ms |
| new (FULL_AND_PIECEWISE) | 35.66 | 17.24 (arm D) | 18.42 ms |

Sluice's real streaming cost is **~19 ms either way**, consistent with the
independently measured `rs48 − noop` = 15.58 ms. The ~14 ms removed was pure
capture-mode waste. Note the *ratio* to baseline worsens (1.69× → 2.07×)
because vanilla gains from FULL too — the absolute win is real, the relative
gap is now dominated by streaming, which is where the remaining work is.

**Why it was being paid.** Sluice forces `PIECEWISE` so its host sync has a
gap. That is correct on the **fx-splitting** path (V2-Lite, Qwen3), where the
gap exists *because* `moe_forward` is in `splitting_ops`. On V4's **breakable**
path the gap comes from `add_eager`, which segments the capture whatever
`cudagraph_mode` says — so Sluice was buying a gap it already had, and losing
the FULL decode family for the 100 % of steps that are pure decode.

**Correctness status — NOT yet proven.** What is established: the gap fires
(counters above); output is coherent and factually correct; and it matches the
shipping config byte-for-byte on 3 of 5 greedy prompts, with the other two
coherent and semantically equivalent. Both configs diverge from vanilla
*identically* on prompt 1, i.e. the divergence is slots=48 miss behaviour
(`VALID_DIVERGENT`), not the capture mode.

The intended bit-identity gate **failed to run**: at slots=64 every layer is
`static_full`, so router-split armed **0/43** layers — a gate with no gap
cannot detect a swallowed gap — and it OOM'd besides. Entry [9] fixes this
with **slots=62**: the envelope bounds the working set at 8 × top-6 = 48
uniques, so 62 slots give **zero misses** (output must be bit-identical to
vanilla) while 62 < 64 keeps the layer out of `static_full` and router-split
**armed**. Both the new mode and the shipping mode must pass; if the shipping
mode fails, the gate is wrong rather than the mode.

**Scope**: this is not V4-specific. Every model vLLM puts on the
breakable-cudagraph path has been losing its FULL decode graphs to this guard.

---

### [10] Sluice vs vanilla at c=1 and c=8, BOTH models, both baselines
**2026-07-27 · four pods in parallel (`sluice-kx`, `-v2`, `-v4s`, `-v2b`)**

Two baselines per model, because conflating them is what produced the 19.42 ms
error:
* **vanilla_stock** — vLLM defaults. The real "should I use Sluice?" comparison.
* **vanilla_matched** — vanilla carrying the flags Sluice requires. Isolates
  Sluice's own cost from the cost of its configuration.

#### DeepSeek-V4-Flash — TP=4, EP, marlin, fp8 KV, slots=48, FULL_AND_PIECEWISE

| arm | c=1 TPOT | c=1 tok/s | c=8 TPOT | c=8 tok/s |
|---|---|---|---|---|
| vanilla_stock | 8.05 | 121.4 | **9.77–10.00** | **729–791** |
| vanilla_matched (`MAX_BT=8`) | 8.08 | 114.6 | 16.98 | 365.4 |
| **SLUICE slots=48** | **14.17** | **66.4** | **38.60** | **168.5** |

vs stock **1.76× / 3.9×**; vs matched **1.75× / 2.27×**. 43/43 armed,
gap-calls=176000, misses=68369.

#### DeepSeek-V2-Lite — TP=1, triton, slots=48, forced PIECEWISE

| arm | c=1 TPOT | c=1 tok/s | c=8 TPOT | c=8 tok/s |
|---|---|---|---|---|
| vanilla_stock | 3.77 | 262.1 | 7.71 | 1023.6 |
| vanilla_matched | 5.15 | 191.2 | 7.81 | 865.8 |
| **SLUICE slots=48** | **7.97** | **121.1** | **13.74** | **482.5** |

vs stock **2.11× / 1.78×**; vs matched **1.55× / 1.76×**. 26/26 armed,
gap-calls=104000, misses=25540. `cudagraph_mode` logged as **PIECEWISE** — the
guard correctly refused FULL on the fx-splitting path, as predicted.

#### Readings

1. **V2-Lite holds up far better at c=8**: 1.78× vs V4's 3.9× against stock —
   47 % of vanilla throughput vs 22 %. Both offload the same fraction (48 of 64
   local experts resident, 25 % off-GPU), so this is **not** the offload ratio.
   It is V4's 4-rank convoy and per-step coordination.
2. **The envelope cost is a V4 phenomenon.** stock → matched costs V2-Lite
   **0.1 ms** at c=8 and V4 **7.0 ms**, from the same two flags. Consistent with
   launch overhead: 43 layers × 256 experts × 4 ranks vs 26 × 64 on one GPU.
3. **Sluice's overhead scales with concurrency on V4** (6.1 ms at c=1 → 21.6 ms
   at c=8) but is nearly flat on V2-Lite (2.8 → 5.9 ms). This is the
   PCIe-per-unique-expert term: a step routes up to `tokens × top_k` experts, so
   the working set grows with the batch. **It refutes my pre-registered
   prediction** that Sluice would look *relatively worse* at c=1 because the
   host sync is a fixed per-step toll — the bandwidth term dominates the sync
   term, and it grows with batch size. Recorded as a failed prediction.

#### Caveats, stated not buried

* **V4 Sluice at c=8 measured 35.66 ms (entry [8]) and 38.60 ms here**, same
  pod, different server instances. Quote it as **36–39 ms**, not a point value.
* **gpu-util is not matched between stock and Sluice arms**: uncapped vanilla
  needs ≥ 0.75 ("No available memory for the cache blocks" below that) and the
  Sluice arms need ≤ 0.55 (slot buffers). No single value serves both. The
  mismatch is demonstrably harmless — stock measures 9.83 at 0.55 (ledger),
  9.81 at 0.75 (entry [7]), 9.77–10.00 at 0.75 (here) — but it is a
  cross-config comparison, not a strictly matched one.

---

### [13] EP OFF cannot widen the envelope — it is MEMORY-bound
**2026-07-27 · V4-Flash, TP=4, EP disabled, `sluice-kx`**

With EP on, each rank holds 256/4 = 64 experts so `slots <= 64` and the
envelope caps at `64//6 = 10`. EP off gives each rank a TP-sliced copy of all
256 experts, so slots could in principle reach ~192 → envelope 32. The ledger
measured one such point and dismissed it ("not a free win: at c=8 it is worse,
54.81 vs 44.83") — but that predates both corrections here: unmatched vanilla
AND forced PIECEWISE. Re-measured with a matched baseline and FULL graphs.

| arm | c=8 | c=32 |
|---|---|---|
| EPoff vanilla_matched (`MAX_BT=32`) | **10.34 ms / 717 tok/s** | 32.57 ms / 786 tok/s |
| EPoff SLUICE slots=192 (envelope 32) | **OOM** | **OOM** |

```
torch.OutOfMemoryError: Tried to allocate 65.34 GiB
```

**Two findings.**

1. **Widening the envelope really does recover the batch-cap penalty — for
   vanilla.** `MAX_BT=32` measures **10.34 ms** against `MAX_BT=8`'s 16.98 ms
   (entry [10]) and stock's ~9.9. So the +7 ms attributed to the cap in entry
   [7] is real and is recovered when the cap is lifted. The mechanism is
   confirmed from both directions.

2. **Sluice cannot get there.** EP off does not reduce per-rank expert memory
   (each rank holds 140 GiB/4 ≈ 35–46 GiB either way — EP splits experts
   *whole*, TP splits them *sliced*, same total). Slot buffers for 192 of 256
   experts need **65.34 GiB on top of the weights** — 102 GiB against 79
   available. Scaling to what fits (~34 GiB of buffers ⇒ slots ≈ 96) leaves
   ~1 GiB for KV; dropping to slots=64 to make room yields envelope
   `64//6 = 10`, barely better than the 8 already in use.

⇒ **The envelope is bounded by GPU memory, not by policy.** On a fixed GPU
count you cannot buy slots without evicting the weights the slots exist to
serve. No code change reaches this.

**The way out is more ranks, not more slots** — and specifically **DP / wide-EP**
(llm-d's `wide-ep-lws` shape: attention data-parallel with TP=1 per rank,
experts EP-sharded across many ranks). The envelope is enforced **per rank**,
so under TP all ranks share one 8-token batch and 8 is a *global* ceiling;
under DP=N each rank gets its own 8, and the aggregate is 8×N. Wide-EP does not
widen the per-rank envelope — it stops the per-rank envelope from being a
global throughput ceiling.

Tension to keep honest: wide EP also **erodes the reason to offload**. At
EP=32 each rank holds 8 experts and there is little left to stream. Sluice's
value is highest at *narrow* EP (serve V4 on 2 GPUs where vanilla OOMs), which
is the opposite regime. The two are partly competing answers to the same
problem and combine only when GPU-constrained *and* throughput-hungry.

**Next, and cheap**: Sluice has a DP path (`_dp_mode`, offloader.py:740-759)
that hooks the modular kernel's post-dispatch seam instead of the pre-dispatch
`apply`, because under DP the hook would otherwise stream for its own rank's
routing while the kernel computes the gathered token set. It also disables the
GEMM-graph path and degrades SLRU. **Whether router-split arms at all under
`_dp_mode` is unverified** — that is the first thing to check, before any
wide-EP performance claim.

---

### [11] THE CORRECTNESS GATE — fourth design, and it finally works
**2026-07-27 · TP=8, slots=32 (full residency), `SLUICE_RS_FORCE_ARM=1`**

Three earlier designs could not test what they were built to test, each for a
different reason, all recorded rather than quietly retried:

| attempt | why it could not work |
|---|---|
| slots=64, TP=4 | every layer `static_full` ⇒ router-split armed **0/43**. No gap exists to be swallowed. |
| slots=62, TP=4 | armed, but the cache evicts *between* steps so misses are unavoidable; **both** configs legitimately diverge. My "zero misses" prediction was simply wrong — misses are not bounded by the per-step working set. |
| slots=64 + FORCE_ARM, TP=4 | **OOM, 45.9 GiB**: slot buffers sit on top of the loaded weights. Infeasible in 80 GiB at any gpu-util. |

TP=8 fixes the memory: with EP each rank holds 256/8 = **32** experts, so
slots=32 *is* full residency at ~23 GiB. Same breakable path, same FULL
capture — only the size changed.

**Measured** (greedy, temperature 0, 5 prompts, 24 tokens):

| arm | prompts 1–4 | prompt 5 | armed | misses | mode |
|---|---|---|---|---|---|
| **GATE_FULL_AND_PIECEWISE** | **byte-identical to vanilla** | diverges | 43/43 | **0** | FULL_AND_PIECEWISE |
| **GATE_PIECEWISE (control)** | **byte-identical to vanilla** | diverges | 43/43 | **0** | PIECEWISE |

`misses=0` confirms the design finally holds: everything resident, nothing
streamed, map static.

#### Three conclusions

**1. The FULL change is exonerated.** The shipping config diverges identically,
so the divergence is not caused by it. The pre-registered rule applies: *when
the control fails, the gate is not indicting the new mode.* FULL_AND_PIECEWISE
is **as correct as forced PIECEWISE** — which is the honest bar for shipping it,
and is not the same claim as "bit-identical to vanilla".

**2. The map is definitively NOT frozen under FULL capture.** This is the
strongest correctness evidence in the campaign: a swallowed D2H sync would
freeze routing and corrupt *every* prompt, yet **4 of 5 are byte-identical to
vanilla**. Combined with live gap counters (entry [8]: 54 000 calls, 31 677
misses), the mechanism is settled.

**3. Router-split is NOT bit-identical to vanilla, even at full residency with
zero misses.** Nothing streams in this configuration, so this is not a
streaming artifact — it is the **restructuring**. Router-split replaces the
fused `vllm::moe_forward` with separate gate → select → gap → `fused_experts`
calls; the different kernel path changes FP accumulation enough to flip a token
sitting on a decision boundary, after which continuations diverge entirely.
(Prompt 5 is visibly such a boundary: it also flipped between TP=4 and TP=8 in
*vanilla*.)

**This corrects a claim the campaign carried loosely.** "Sluice is bit-identical
at full residency" holds for the **classic streaming path** — identity map,
stock fused op untouched. It does **not** hold for the **router-split path**,
which is a different computation by construction. The two were being treated as
one guarantee. Any future correctness gate for router-split must compare
against *router-split*, not against vanilla.

**Recommendation**: keep `SLUICE_ALLOW_FULL_CG` as an explicit opt-in and
document the −28 % win; do **not** flip it to default on the strength of five
prompts. Defaulting deserves a task-level eval (accuracy on a real benchmark),
comparing router-split-FULL against router-split-PIECEWISE — not against
vanilla, since that comparison can never pass.
