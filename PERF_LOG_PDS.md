# pds-perf log — Sluice on DeepSeek, V2-Lite first

Worktree `perf/pds-perf`. Pods `sluice-v2` (1 GPU) and `sluice-kx`, run in
parallel. Successor to `sluice-v4-perf`, whose performance model was disproved
on 2026-07-27.

---

### [1] H1 — can the FULL-cudagraph win transfer to V2-Lite? **No, and not for the predicted reason.**
**2026-07-27 · CLOSED as a lever · the hypothesis remains UNTESTED behind a defect**

- **Hypothesis**: V4 gained −28 % by keeping vLLM's FULL decode graphs. V2-Lite
  cannot today — it is on the fx-splitting path where the guard correctly
  refuses. But `VLLM_USE_BREAKABLE_CUDAGRAPH` is **user-settable for any model**
  (vLLM only *auto*-enables it for V4/MiniMax,
  `vllm/config/vllm.py:1113-1127`), which would give mode `NONE`, empty
  `splitting_ops`, an `add_eager` gap — exactly what `SLUICE_ALLOW_FULL_CG=1`
  needs.
- **Pre-registered**: ~50/50. Breakable disables torch.compile entirely; on V4
  compile was already off so FULL was pure gain, but V2-Lite might depend on
  Inductor. Rule: **≥5 % ⇒ pursue, worse ⇒ close.**
- **Design**: 2×2 so "breakable helps" can be told apart from "Sluice gains
  from breakable" — the arm the prediction hinged on.

**Measured** (`bench serve`, random 32/200, c=8, warm + 3 repeats):

|  | fx + PIECEWISE | breakable + FULL |
|---|---|---|
| **vanilla** | 7.93 ms (925/916/919 tok/s) | **7.79 ms** (938/934/939) |
| **SLUICE slots=48** | 12.68 ms (561/563/558) | **38.39 ms** (196/183/193) |

Both Sluice arms confirmed 26/26 layers armed; arm B logged
`mode: NONE` + `cudagraph_mode: FULL_AND_PIECEWISE`, so the configuration
genuinely engaged.

**Verdict: closed. And my predicted mechanism was wrong.**

I first read this as "breakable disables Inductor, and that dwarfs FULL's
gain". **Arm C refutes that**: vanilla loses Inductor too and is *unaffected*
(7.79 vs 7.93 — marginally faster). Vanilla is indifferent to the capture path;
only Sluice degrades, and by 3×.

**The mechanism is a cache defect, and the counters name it:**

| arm | gap-calls | misses | misses/call |
|---|---|---|---|
| A fx + PIECEWISE | 188 000 | 64 238 | 0.34 |
| B breakable + FULL | 188 000 | **640 657** | **3.41** |

Identical gap-call counts, **10× the misses** — more misses than calls. The slot
cache retains almost nothing on this path; Sluice re-streams nearly every routed
expert every step. That is a bug, not a trade-off.

⇒ **The hypothesis was never actually tested.** Whether FULL decode graphs help
V2-Lite is still open, behind that defect. Arm C shows the capture-side ceiling
is at least vanilla parity, so fixing the thrash would reopen a real lever.

**Open question for whoever picks this up**: why does Sluice's slot cache
thrash on the breakable path for a non-V4 model? Candidates: the fast-hit path
not engaging, the map write not landing where the GEMM reads it, or per-step
cache state being reset. `SLUICE_RS_FAST_HIT` was on in both arms.

---

### [2] H2 — decompose V2-Lite's gap, and the slot trade curve
**2026-07-27 · the first curve this project can actually read**

Every previous slot sweep ran against an unmatched baseline. This one has a
flag-matched vanilla control run **first and last**: **8.02 ms** and **8.08 ms**
— **1 % apart**, so there is no drift and every delta below stands as measured.

**Measured** (same pod, same session, warm + 3 repeats each):

| slots | offloaded | TPOT | tok/s | vs vanilla | misses |
|---|---|---|---|---|---|
| 48 | 25 % | 12.77 ms | 557 | **+59 %** | 65 039 |
| 56 | 12.5 % | 10.75 ms | 668 | **+34 %** | 21 247 |
| 62 | 3 % | 10.01 ms | 723 | **+25 %** | 4 705 |
| vanilla (64) | 0 % | **8.02–8.08 ms** | 918 | — | — |
| *no-op floor @48* | *(invalid output)* | *6.30 ms* | *1148* | *−21 %* | *0* |

**Two structural numbers:**

1. **Sluice has a floor of ~2 ms/step.** At slots=62 the cache misses on only
   2.5 % of gap calls and the GEMM is nearly vanilla's size, yet it is still
   **+1.99 ms**. That is the irreducible mechanism cost — D2H sync, map writes,
   the split ops — and it does **not** depend on how much is offloaded.
2. **Everything above the floor is miss-driven.** 62 → 48 slots buys 22 more
   points of offload for **+2.76 ms**.

**A correction to my own instrument.** The no-op floor is **faster than
vanilla** (6.30 vs 8.02). That is not Sluice winning: under `RS_NOOP` nothing
streams, the map never updates, tokens routed to absent experts contribute
zero, and the GEMM runs over 48 slots instead of 64 experts. It does **less
work**. So "structural = noop − matched" is **not a valid subtraction**, and I
had used exactly that framing earlier today on V4 — where it was defensible
only by luck, because the floor happened to land on top of matched vanilla
(29.25 vs 29.11) and the work-reduction effect was invisible. The floor also
rises with slot count, so it is not a constant across this curve either. The
vanilla-relative column above avoids it entirely.

**What this gives the project**: the trade curve nobody had drawn — *how much
host-RAM offload do you want, and what does each increment cost?* On V2-Lite,
3 % offload costs 25 % throughput; 25 % offload costs 39 % throughput.

**Next**, in order: (a) attack the ~2 ms floor — that is the sync-side flags
(`RS_STAGED`, `LAZY_SYNC`, `HOOK_LITE`, `LAZY_STEP`), none re-measured since
the retraction; (b) the breakable-path cache defect from entry [1]; (c) repeat
this curve on V4, where misses run 38.8 % and the floor should be larger.

---

### [3] Qwen3-30B-A3B — a third model, and a memory wall Sluice does not hit on DeepSeek
**2026-07-28 · TP=1, 1×H100, triton, bf16**

A different point in the design space from both DeepSeek models:

| | layers | experts | top-k | dtype | weights |
|---|---|---|---|---|---|
| V2-Lite | 26 | 64 | 6 | bf16 | ~29 GiB |
| V4-Flash | 43 | 256 | 6 | fp8 | ~149 GiB (TP=4 ⇒ ~37/rank) |
| **Qwen3-30B-A3B** | **48** | **128** | **8** | **bf16** | **~57 GiB on ONE card** |

**Vanilla, c=1/3/6** (`bench serve`, random 32/200):

| arm | c=1 | c=3 | c=6 |
|---|---|---|---|
| vanilla_stock | 4.74 ms / 208.8 | 8.24 / 360.8 | 10.58 / 560.6 |
| vanilla_stock (drift ctrl) | 4.44 / 222.8 | 7.95 / 373.7 | 10.34 / 573.1 |
| vanilla_matched `MAX_BT=8` | 5.67 / 173.8 | 8.47 / 347.5 | 10.69 / 530.9 |
| vanilla_matched `MAX_BT=6` | 5.59 / 175.6 | 8.45 / 346.0 | **9.32 / 561.9** |
| **SLUICE slots=64** | **OOM** | — | — |
| **SLUICE slots=48** | **OOM** | — | — |

Drift control: 2–7 % between the two stock arms, tightest at high c. Both
vanilla arms logged `VLLM_COMPILE` + the expected cudagraph mode, so Qwen3 is
on the **fx-splitting path** like V2-Lite — entry [8]'s FULL win does not apply,
as predicted.

#### Finding 1 — Sluice's peak is `weights + slot cache`, and it blocks this model

Each slot costs **0.45 GiB** (58 GiB of experts ÷ 128). So:

| slots | slot cache | peak | result |
|---|---|---|---|
| 64 (50 % offload) | 29 GiB | ~86 GiB | OOM |
| 48 (37.5 % offload) | 22 GiB | ~79 GiB | **OOM, 318 MiB short** |

The slot buffers are allocated **while the full model is still resident**. V4
never exposes this because TP=4 divides the weights first (~37 GiB/rank); at
TP=1 with 57 GiB of weights there is no room.

Cutting slots far enough to fit is not a workaround: the envelope is
`slots // top_k` and **top-8** makes it tight, so anything small enough to fit
drops the envelope below 6 and c=6 stops taking the split path — the arm would
no longer be comparable across the concurrencies under test.

⇒ **The single-GPU, near-capacity MoE is precisely the case Sluice exists to
serve, and its load-time peak is what blocks it.** This deserves to be treated
as a limitation of the memory model, not a tuning problem. Entry [3c] re-runs
everything at TP=2 (28.5 GiB weights + 14.5 GiB slots per rank ⇒ ~43 GiB peak,
envelope back to 8).

#### Finding 2 — the envelope cost SHRINKS with concurrency, on every model

| model | flag cost at low c | at high c |
|---|---|---|
| Qwen3-30B | **+0.93 ms** (c=1) | **+0.11 ms** (c=6) |
| V2-Lite | +1.38 ms (c=1) | +0.10 ms (c=8) |

The `MAX_BT` cap bites where chunked prefill dominates the measurement and is
nearly free once decode dominates. The V4 campaign booked a **flat +7 ms** to
that cap; this says the shape was never that simple, and any future envelope
claim must state the concurrency it was measured at.

Also: `MAX_BT=6` at c=6 measures **9.32 ms — faster than stock's 10.58**.
Capping at exactly the concurrency keeps prefill chunks out of decode steps,
lowering TPOT at the cost of TTFT. So "matched" is not automatically a *worse*
baseline than stock; here it is the harder one to beat.

---

### [3c/3d] Qwen3-30B-A3B at TP=2 — the complete c=1/3/6 comparison
**2026-07-28 · 2×H100, triton, bf16, slots=64 of 128 (50 % offloaded)**

TP=2 is where Sluice fits: per-rank weights ~28.5 GiB instead of 57.

**Measured** (`bench serve`, random 32/200; drift control within **1 %**):

| arm | gpu-util | c=1 | c=3 | c=6 |
|---|---|---|---|---|
| vanilla_stock | 0.90 | 3.92 ms / 252.8 | 5.59 / 530.4 | 7.09 / 834.7 |
| vanilla_stock *(drift ctrl)* | 0.90 | 3.96 / 249.9 | 5.60 / 529.0 | 7.14 / 829.0 |
| vanilla_matched | 0.90 | 5.92 / 167.4 | 5.94 / 493.3 | 7.25 / 779.2 |
| **vanilla_matched** | **0.60** | **6.17 / 160.6** | **6.60 / 444.6** | **7.30 / 773.6** |
| **SLUICE slots=64** | **0.60** | **8.28 / 118.4** | **13.07 / 222.7** | **22.10 / 258.8** |

The two matched controls (0.90 vs 0.60) agree to within 0.25–0.66 ms, so
gpu-util is not confounding the Sluice comparison.

**Sluice vs its same-setting control:** **1.34× / 1.98× / 3.03×**
**vs stock:** 2.11× / 2.34× / **3.12×** — i.e. **47 % / 42 % / 31 %** of vanilla
throughput.

#### Sluice degrades hard with concurrency on this model — top-8 is why

`gap-calls=280 000, misses=291 991` ⇒ **1.04 misses per gap call**, against
V2-Lite's 0.35 at slots=48. With top-8 routing, a c=6 step wants up to
6 × 8 = **48 distinct experts** and only 64 of 128 are resident, so the cache
thrashes. Ranking at high concurrency across all three models:

| model | top-k | Sluice vs stock at high c | misses/call |
|---|---|---|---|
| V2-Lite | 6 | 1.78× (c=8) | 0.35 |
| **Qwen3-30B** | **8** | **3.12× (c=6)** | **1.04** |
| V4-Flash | 6 | 3.9× (c=8) | 0.39 |

#### The gpu-util trap — a design property, not cluster trivia

Sluice OOMed at gpu-util 0.90 at **both** TP=1 (slots 64 and 48) and TP=2. The
peak is **weights + KV + slot cache**: vLLM sizes the KV cache to fill the
budget *before* Sluice allocates slot buffers, so the slot cache is charged on
top of a budget already spent. TP=2 at 0.90: 28.5 + ~40 + 14.5 ≈ 83 GiB against
a 79 GiB card.

**This is why the V4 runs needed 0.55 for Sluice against 0.75 for vanilla** — a
workaround previously recorded without its reason. The operator must hand-lower
`--gpu-memory-utilization` by an amount proportional to `slots × per-expert
size`, and the only feedback is a bare CUDA OOM that never mentions Sluice.

**And the saving it blocks is real.** KV cache actually allocated:

| arm | gpu-util | KV tokens |
|---|---|---|
| vanilla_stock | 0.90 | 842 672 |
| vanilla_matched | 0.90 | 913 392 |
| **SLUICE slots=64** | **0.60** | **988 336** |

Sluice ends up with **more** KV headroom than vanilla despite a far lower
budget — its steady-state memory saving works exactly as designed. **The OOM is
purely load-time allocation ordering**, and it is what stopped Sluice running at
TP=1, the configuration where the saving would have mattered most. That is a
fixable bug and it should outrank further tuning.

---

### [4] ROOT CAUSE — vLLM's memory profiler does not count Sluice's slot cache
**2026-07-28 · Qwen3-30B-A3B, TP=2, vLLM DEFAULT `gpu_memory_utilization`**

Run with **no `--gpu-memory-utilization` on any arm**, to remove the objection
that hand-tuned memory settings shaped the comparison. Default is 0.9, and the
vanilla arms reproduce the explicit-0.90 numbers to within 1 %:

| arm | c=1 | c=3 | c=6 |
|---|---|---|---|
| vanilla_stock | 3.91 / 252.8 | 5.60 / 529.5 | 7.17 / 826.0 |
| vanilla_stock *(drift ctrl)* | 3.96 / 250.1 | 5.62 / 527.7 | — |
| vanilla_matched | 6.59 / 150.3 | 6.54 / 448.1 | 7.10 / 793.8 |
| **SLUICE slots=64** | **OOM** | — | — |
| **SLUICE slots=96** | **OOM** | — | — |

#### The reproducer, in one line per arm

```
vanilla_matched   Available KV cache memory: 43.40 GiB
SLUICE slots=64   Available KV cache memory: 70.29 GiB
SLUICE slots=96   Available KV cache memory: 70.58 GiB
```

**slots=64 → 96 adds ~6.5 GiB of live slot buffers and moves the reported
available KV by 0.29 GiB.** The slot cache is **invisible** to vLLM's memory
accounting. vLLM then sizes KV at ~70 GiB on a 79 GiB card while 15–21 GiB is
genuinely held, and dies allocating it:

```
gpu_worker.py:710, in initialize_from_config
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.46 GiB
```

#### Lifecycle — the ordering is fine; the accounting is not

```
gpu_worker.load_model -> gpu_model_runner.load_model(5204)
                      -> get_offloader().post_init()      <- slots allocated HERE
gpu_worker.determine_available_memory(430)                <- profiling, AFTER
gpu_worker.initialize_from_config(695)                    <- KV alloc -> OOM
```

`post_init` runs **before** profiling (`gpu_model_runner.py:5381`), so the
buffers exist when vLLM measures. They are simply not attributed. My first
explanation — that `empty_cache` hid them — was wrong for the right reason:
`torch.accelerator.empty_cache()` (`offloader.py:636`) proves freed expert
weights are *returned to the driver*, not that live slot buffers are *counted*.

#### What this single bug explains

- **Why V4 needed `gpu-util 0.55` against vanilla's `0.75`** — the operator was
  hand-compensating for uncounted slots. Recorded in the agent brief as an
  environment quirk with no reason attached; it is a design defect.
- **Why Qwen3 OOMed at TP=1 at every slot count tried** (64 and 48).
- **Why Sluice at 0.60 ended up with MORE KV (988 336 tokens) than vanilla at
  0.90 (947 984)** — the profiler over-credits it there too, just not fatally.
- **Why the failure is a bare CUDA OOM that never mentions Sluice.**

#### Fix direction

Make the slot bytes visible to the profiler: allocate the cache before vLLM's
baseline snapshot, or add `sum(slot.nbytes)` to vLLM's non-torch memory
accounting. Sluice already computes this figure for its own operator log.

**Until fixed, Sluice cannot run at vLLM's default `gpu_memory_utilization`** —
a reasonable thing for a user to expect, and the reason every Sluice benchmark
in this project has carried a hand-tuned memory flag.

**Priority**: this outranks further tuning. It blocks the single-GPU
near-capacity MoE — precisely the case Sluice exists to serve — and the
steady-state saving underneath it is real (entry [3d]: Sluice genuinely frees
expert memory and ends up with more KV headroom than vanilla).

---

### [5] FIX — report the slot cache to vLLM's profiler. Verified, and it unlocks slots=96.
**2026-07-28 · Qwen3-30B-A3B, TP=2, vLLM DEFAULT `gpu_memory_utilization`**

**Change**: the plugin's `load_model` wrapper adds the slot-cache bytes to
`runner.model_memory_usage`. That seam runs **after** `post_init` (slots
allocated, `gpu_model_runner.py:5381`) and **before**
`determine_available_memory` (`gpu_worker.py:430`) — exactly the gap the
buffers fell through. Sluice already computed the figure for its operator log;
it simply never told vLLM. Two edits: `offloader.py` publishes
`self.slot_vram_bytes`; `plugin.py` adds it and logs the correction.

**Pre-registered before the run**: the reported available KV must DROP BY
ROUGHLY THE SLOT SIZE *and* the two slot counts must SEPARATE — starting
without OOM was explicitly declared insufficient, since that could be luck.

| slots | slot cache | available KV before | **after** | predicted |
|---|---|---|---|---|
| 64 | 13.50 GiB | 70.29 GiB | **57.08 GiB** | ~56–57 ✅ |
| 96 | 20.25 GiB | 70.58 GiB | **50.33 GiB** | ~50 ✅ |
| **Δ** | **6.75 GiB** | **0.29 GiB** ❌ | **6.75 GiB** ✅ | |

The arms now separate by **exactly** the slot-size difference. Both start at
default utilization, where both OOMed before.

**KV recovered is larger than the workaround gave**: 1 246 928 tokens at
slots=64 against 988 336 under the hand-tuned `gpu-util 0.60`. Correct
accounting beats guessing headroom.

**Performance is unchanged by the fix** (8.37/13.39/23.01 vs 8.28/13.07/22.10
at util 0.60, within 4 %; misses 292 806 vs 291 991). The workaround was
costing correctness of accounting, not speed — which is why it survived so long.

#### slots=96, previously unreachable, is a large win

| c | slots=64 | slots=96 | throughput gain |
|---|---|---|---|
| 1 | 8.37 ms / 117.3 | 8.48 / 116.5 | — |
| 3 | 13.39 / 217.3 | **10.39 / 281.1** | **+29 %** |
| 6 | 23.01 / 248.4 | **13.21 / 426.1** | **+72 %** |

**Misses 292 806 → 39 861 (7.3× fewer.)** With top-8 routing, 96 of 128 experts
resident stops the thrash that made c=6 so bad. Against vanilla at c=6 Sluice
goes from **3.21× → 1.84×**, and 30 % → 52 % of vanilla throughput.

#### Complete table — TP=2, default utilization, no hand-tuned flags anywhere

| arm | c=1 | c=3 | c=6 |
|---|---|---|---|
| vanilla_stock | 3.91 / 252.8 | 5.60 / 529.5 | 7.17 / 826.0 |
| vanilla_matched | 6.59 / 150.3 | 6.54 / 448.1 | 7.10 / 793.8 |
| SLUICE slots=64 | 8.37 / 117.3 | 13.39 / 217.3 | 23.01 / 248.4 |
| **SLUICE slots=96** | **8.48 / 116.5** | **10.39 / 281.1** | **13.21 / 426.1** |

vs stock — slots=96: **2.17× / 1.86× / 1.84×**; slots=64: 2.14× / 2.39× / 3.21×.

#### Consequences

- **Every Sluice benchmark in this project used a hand-lowered
  `--gpu-memory-utilization`.** Those runs were not wrong on speed, but they
  were all working around this defect, and any of them that reported an OOM as
  a capacity limit should be revisited — starting with entry [3]'s claim that
  Sluice cannot fit Qwen3 at TP=1.
- The V4 `0.55` vs vanilla `0.75` asymmetry has a cause and a fix.
- **Slot sizing was being tuned against a broken memory budget**, so earlier
  slot sweeps were exploring a smaller space than they appeared to.

---

### [6][7] Flag minimisation, and a WITHDRAWN conclusion: Qwen3 DOES fit at TP=1
**2026-07-28 · Qwen3-30B-A3B, all at vLLM defaults except where noted**

#### What can be dropped, measured rather than assumed

| flag | verdict | evidence |
|---|---|---|
| `--gpu-memory-utilization` | **droppable** | today's memory-accounting fix |
| `--max-model-len` | **droppable** | 8.36/10.39/13.29 vs 8.48/10.39/13.21 with it |
| `--trust-remote-code` | **droppable** | same run |
| `--max-num-batched-tokens` | **KEEP** | only removable by dropping router-split — costs 3–4× |
| `-cc.cudagraph_mode=PIECEWISE` | **KEEP** on Qwen3 | refusal verified with the actual error |
| `--moe-backend` | **KEEP** | must honor `expert_map` |

Minimal working command, seven flags down to four:

```bash
export SLUICE_SLOTS=96 SLUICE_ROUTER_SPLIT=1 SLUICE_PIECEWISE=1 SLUICE_RS_FAST_HIT=1
vllm serve Qwen/Qwen3-30B-A3B \
  --tensor-parallel-size 2 --moe-backend triton \
  --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE
```

#### The envelope is the price of the fast path — quantified

Dropping `--max-num-batched-tokens` requires dropping `SLUICE_ROUTER_SPLIT`,
because the envelope check is gated on it. The classic path then runs at vLLM's
default 8192:

| config | c=1 | c=3 | c=6 |
|---|---|---|---|
| router-split, cap 8 | 8.48 / 116.5 | 10.39 / 281.1 | **13.21 / 426.1** |
| classic, no cap | 38.33 / 26.1 | 40.17 / 73.9 | **40.03 / 148.1** |

**3–4× worse.** The cap costs far less than the fast path is worth. Note the
shapes differ: classic TPOT is flat across concurrency (38→40→40) while
router-split degrades (8.5→10.4→13.2), because router-split can never batch
more than `slots // top_k` = 8 decode tokens per step while classic is
unbounded. A crossover at higher concurrency is likely and unmeasured.

#### WITHDRAWN: "Sluice cannot fit Qwen3 at TP=1"

Entry [3] concluded that, and framed it as a limitation of the memory model —
*"the single-GPU near-capacity MoE is exactly the case Sluice exists for, and
its load-time peak is what blocks it."* **That is wrong.** It was measured
before the memory fix, when vLLM reported 70.29 GiB available on a 79 GiB card
because it could not see the slot cache. Post-fix:

| TP=1, 1×H100 | c=1 | c=3 | c=6 |
|---|---|---|---|
| vanilla_stock (entry [3]) | 4.74 / 208.8 | 8.24 / 360.8 | 10.58 / 560.6 |
| **SLUICE slots=64** | **7.96 / 121.9** | **12.86 / 224.9** | **30.06 / 191.0** |
| ratio | 1.68× | 1.56× | 2.84× |

Sluice reports **41.91 GiB free for KV on a single card**, against ~14 GiB for
vanilla (57 GiB of weights resident) — roughly **28 GiB of GPU memory freed on
one GPU**. That is Sluice's actual value proposition, and until now this
campaign had never measured it: every prior number came from configurations
where vanilla already fit comfortably.

At c=1, TP=1 is **faster than TP=2** (7.96 vs 8.36 ms) — no per-step
tensor-parallel communication.

c=6 degrades badly (2.84×) for the familiar reason: **245 884 misses = 0.88 per
gap call** with only 64 of 128 experts resident under top-8 routing. Entry [8]
sweeps slots 96/112 at TP=1, where the freed memory leaves headroom.

**Lesson**: the OOM was read as a capacity limit and written up as a design
property. It was a bug in one line of accounting. Any conclusion of the form
"Sluice cannot fit X" recorded before 2026-07-28 should be re-run.

---

### [8] THE TRADE CURVE ON ONE GPU — Qwen3-30B-A3B, TP=1
**2026-07-28 · 1×H100, vLLM defaults, same-pod vanilla control**

The configuration Sluice exists for, measurable only after the memory fix.
All arms on one pod, one session; vanilla re-measured here rather than reused
(the entry [3] TP=1 vanilla came from a different pod and differs by 17 % at
c=1 — 4.74 vs 5.55 ms — which is why cross-pod ratios were restated).

| arm | KV free | **GPU freed** | c=1 | c=3 | c=6 | misses |
|---|---|---|---|---|---|---|
| vanilla | 12.56 GiB | — | 5.55 / 178.6 | 8.45 / 351.6 | 10.39 / 570.1 | — |
| SLUICE slots=112 | 22.19 | **+9.6 GiB** | 8.28 / 118.5 | 10.52 / 278.0 | **13.91 / 406.6** | 18 443 |
| SLUICE slots=96 | 28.94 | **+16.4 GiB** | 8.07 / 121.2 | 11.01 / 264.7 | 16.56 / 341.8 | 47 911 |
| SLUICE slots=64 | 41.91 | **+29.4 GiB** | 7.96 / 121.9 | 12.86 / 224.9 | 30.06 / 191.0 | 245 884 |

vs vanilla at c=6: **1.34× / 1.59× / 2.89×** (71 % / 60 % / 34 % of throughput).

#### Two structural readings

**1. At c=1 the slot count is irrelevant** — 7.96 / 8.07 / 8.28 ms across a
75 % → 87.5 % residency range. Single-stream cost is the per-step *mechanism*
(D2H sync, map write, split ops), not the cache. This matches V2-Lite's ~2 ms
floor: slots only start to matter once concurrency pushes the working set past
them.

**2. top-8 sets the shape.** A c=6 step routes up to 6 × 8 = 48 distinct
experts. 64 slots thrash (0.88 misses/call), 112 barely miss (0.066/call). The
same offload *fraction* costs far more on a high-top-k model than on the top-6
DeepSeeks — which is why Qwen3 looked worse than V2-Lite at equal residency.

#### What this is worth as a product statement

On a single H100, Qwen3-30B-A3B: **give back ~10 GiB of GPU memory for ~34 %
throughput, or ~29 GiB for ~66 %.** Every 6–7 GiB reclaimed costs roughly
15–25 % throughput at c=6, and the miss count tracks it exactly.

This is the first time the campaign has priced the *fitting* trade rather than
measuring throughput in configurations where vanilla already fit comfortably.

---

### [9] RETRACTION — every Qwen3 Sluice number today was measured on garbage output
**2026-07-28 · confirmed against issue #4**

Issue #4: *"Router-split produces silently wrong output whenever Inductor
compilation and CUDA-graph capture are both active."* Every Qwen3 (and
V2-Lite) Sluice run in this log used exactly that combination —
`CompilationMode.VLLM_COMPILE` + `cudagraph_mode PIECEWISE` +
`SLUICE_ROUTER_SPLIT=1`. This worktree branched before the refusal commit
(`bc41a31`), so nothing failed loudly.

**Verified directly.** Vanilla reference vs the exact config the tables were
measured on:

```
REF vanilla   Paris. The capital of the United Kingdom is London. The capital ...
              2, 3, and 5. The number 30 is the product of the
              212°F. When the temperature of a pot of water is 212°F

A  suspect    the and and and and and and and and and and and and and and and
              books the the the the the the the the the the the the the the the
              and and and and and and and and and pandemic pandemic pandemic
```

Degenerate repetition — issue #4's failure mode exactly.

#### What is retracted

- **All Qwen3 Sluice numbers**: the c=1/3/6 tables at TP=1 and TP=2, the
  single-GPU trade curve (+9.6 / +16.4 / +29.4 GiB vs 1.34× / 1.59× / 2.89×),
  the slots=64 → 96 "+72 % throughput" claim, and entries [3]–[8]'s Sluice rows.
- **Entry [2]'s V2-Lite slot curve** is exposed to the same defect and had no
  output gate either. Treat as unverified pending a check.

#### What survives

- **The memory-accounting fix ([5])** — it is verified by vLLM's
  `Available KV cache memory` figures (70.29 → 57.08 GiB at slots=64, 50.33 at
  96, separating by exactly the 6.75 GiB slot-size difference), which is
  allocation behaviour and independent of generated tokens.
- **All V4 results**, including the −28 % FULL_AND_PIECEWISE win: vLLM disables
  Inductor for V4, so it never enters the affected combination. V4 also had a
  greedy-output gate.
- **Every vanilla number** — no Sluice in those processes.

#### The process failure, stated plainly

V2-Lite and V4 both had greedy-output gates in this campaign. **Qwen3 got
none.** I took a new model straight to throughput without a validity check, and
benchmarking with `--ignore-eos` on random tokens is precisely the setup where
garbage output is invisible: the harness records tokens/second whether or not
the tokens mean anything.

Worse, the failure mode **flatters** Sluice — frozen routing means the captured
GEMM stops needing fresh experts, so a broken run can be *faster* than a correct
one. The retracted numbers are therefore likely optimistic, not merely wrong.

**Rule for this agent, going forward: no throughput number from a new model or
a new capture configuration is reportable until a greedy-output check against
vanilla has passed in that exact configuration.**

#### Pending

Arm B (`VLLM_USE_BREAKABLE_CUDAGRAPH=1` — Inductor off, graphs kept, issue #4's
validated workaround) was running when cluster auth expired. It will give both
a correctness verdict and valid replacement numbers.

Note a contradiction still to resolve: entry [1] measured that same breakable
path on V2-Lite at slots=48 as **3× slower with 10× the misses**, while issue #4
reports it at slots=60 with **3.1 % variance**. Either the effect is strongly
slot-dependent, or there is a second problem in the breakable path.
