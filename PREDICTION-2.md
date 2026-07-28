# H-KERNEL entry [3] — PREDICTION, written before any run

**Question.** Entries [1] and [2] proved the *cut* is not the cost: at **0
breaks** — one undivided graph, vanilla's shape — router-split still runs
**33.85 ms** against vanilla's **9.83 ms**. What is the remaining **24.02 ms**?

**Two surviving candidates** (C3 "fewer cuts" is closed):

- **C1 — the clone.** `_sluice_stream_gap` ends in `hidden_states.clone()`,
  43×/step.
- **C4 — the restructuring.** Router-split replaces vanilla's single fused
  `vllm::moe_forward` with five ops (gate · select · gap · fused_experts ·
  shared). That is what router-split *exists to do*, so if this is the cost it
  is a floor, not a bug.

**Instrument.** `SLUICE_NO_CLONE`: `0`=clone (normal), `1`=return the input
(no alloc, no copy), `2`=`empty_like` (alloc, no copy — separates allocation
from the copy). Run under `SLUICE_RS_NOOP=1`, where outputs are invalid by
design and the gap body already does nothing; then at **N=0 breaks +
NO_CLONE=1** the gap op is a **pure passthrough** and the only thing left
distinguishing the arm from vanilla is the restructuring itself.

## The arithmetic C1 has to beat — stated first, so the result is scored

Decode at c=8 with `--max-num-batched-tokens 8`: `hidden_states` is `[8, H]`,
H≈7168, bf16 ⇒ **≈114 KiB** per clone. 43 clones ⇒ **≈4.8 MiB/step**. At H100
HBM bandwidth (~3 TB/s, read+write) that is **~3 µs of bandwidth**; even
charging a generous 10 µs of launch overhead per clone gives **~0.43 ms**.

> **C1 cannot plausibly exceed ~0.5 ms of the 24.02 ms.**

There is a second, sharper reason to doubt both C1 and C2 (op dispatch) at
N=0: **a captured graph replays without Python.** At zero breaks there are no
eager segments, so per-step dispatcher work, fake-tensor bookkeeping and
Python call overhead are paid at *capture* time, once — not per step. Anything
that only exists in Python therefore cannot be charged 24 ms per step.

**Prediction, with the decision rule fixed in advance:**

| arm | predicted TPOT | rule |
|---|---|---|
| N=0, clone (known) | 33.85 ms | reference |
| N=0, no-clone | **33.3–33.9 ms** | if it drops **> 3 ms** ⇒ C1 is real, promote it. If it drops **< 1 ms** ⇒ **C1 is dead**, the cost is C4/structural. |
| N=43, no-clone | ~30.0 ms (vs 30.41 known) | cross-check at the normal break count |

**Honest prior: ~85/15 that the clone is NOT the cause.** The bandwidth
arithmetic is off by ~50× and the replay argument says per-step Python is not
even being paid. I am running it anyway because it is one flag and because the
alternative (C4) is the expensive conclusion — it says the cost is intrinsic to
router-split, and I should not reach that without eliminating the cheap
explanation first.

**If C1 dies**, the 24 ms is not explainable by anything in the *gap*, and the
next instrument is not another flag but a **profiler diff**: capture a decode
trace for vanilla and for router-split, and compare kernel-level totals per
step. That answers "what exactly" instead of narrowing by elimination.

**Config:** identical to `PREDICTION.md` (V4-Flash, TP=4/EP, marlin, fp8 KV,
slots=48, MAX_BT=8, PIECEWISE, random 32/200, np=64, c=8, warm + 2 measured
repeats), pod `sluice-kx`. Other agent owns `sluice-g25` — do not touch.

---

# Entry [4] — PREDICTION, written before any run of it

**Question.** Is the router-split cost **per armed layer** (⇒ it is the op
restructuring, C4, and it is a floor) or a **fixed** consequence of being in
this capture mode at all (⇒ something else, and possibly fixable)?

**Instrument.** `SLUICE_RS_LAYERS=N` — arm router-split on only the first N
MoE layers; the rest keep vLLM's stock fused path. Under `SLUICE_RS_NOOP=1`
the classic streaming hook is *also* made a passthrough (gated on both flags,
unreachable in any serving config), so **nothing streams in any arm** and N is
the only variable.

**A vanilla arm runs on this node.** The ledger's 9.83 ms came from a
different instance. Entry [3]'s first arm already shows why that matters:
N=0-breaks measures **28.6 ms here** against entry [2]'s **33.85 ms** — so
entry [2]'s "0 breaks is worse than 1 break" anomaly is looking like a
property of that instance, not of zero-break capture. Cross-instance deltas in
this campaign are worth ~5 ms of noise and must not be quoted as effects.

**Predicted shapes and what each would mean:**

| shape of TPOT vs N | reading |
|---|---|
| **linear** from vanilla at N=0 to full at N=43 | C4 confirmed: cost is per-layer op restructuring. Quantifies it as ms/armed-layer. This is the floor of the current design. |
| **flat then a step** at N>0 | fixed capture-mode effect; the per-layer work is cheap and something global is being paid once. Fixable in principle. |
| **N=0 already ≫ vanilla** | the cost is not router-split at all — it is Sluice merely being *attached* (slot buffers, patched apply, memory layout). Redirects the whole investigation. |

**Honest prior: ~70/30 toward linear.** Every elimination so far has pushed
toward the restructuring. The third outcome is the one I would most want to
catch and the one I have not yet ruled out — which is exactly why the N=0 and
vanilla arms are both in the sweep rather than assumed equal.
