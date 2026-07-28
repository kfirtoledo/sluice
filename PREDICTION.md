# H-KERNEL entry [1] — PREDICTION, written before any run

**Hypothesis.** The 19.42 ms "graph-shape" term (55.5 % of V4's gap to vanilla,
PERF_LOG entry [11]) is dominated by the **number of breakable-cudagraph
segment boundaries** (43 `add_eager` per step). If true, a device-side gap that
needs no CPU round-trip would remove *all* breaks and recover most of it — and
a custom CUDA kernel is justified. If false, the kernel is an expensive wrong
answer and the cost lives somewhere else (the `hidden_states.clone()`, gap-op
dispatch, or capture-region effects that do not scale with segment count).

**Instrument.** `SLUICE_EXTRA_BREAKS=K` (this worktree only): inject K
additional **empty** `cap.add_eager(_sluice_noop_break)` calls per gap. Adds
segment boundaries WITHOUT adding sync, PCIe, map writes or clones — every
other term is held fixed, so d(TPOT)/dK is the pure per-break cost.

Run with `SLUICE_RS_NOOP=1` so the 43 real gaps are also empty: then all
43×(1+K) breaks are uniform and K=0 should reproduce entry [11]'s 29.25 ms
floor. Outputs invalid by design — timing instrument only.

**Arithmetic.** Entry [11] attributes 19.42 ms across 43 breaks = **451.7
µs/break**, but that figure also contains the clone and gap-op dispatch, which
do NOT scale with K. So:

| model | per-break cost | predicted slope (per K = +43 breaks) | predicted TPOT at K=0/1/2 |
|---|---|---|---|
| **H-linear** — segment count dominates | ~452 µs | **+19.4 ms** | 29.3 / 48.7 / 68.1 ms |
| **H-small** — cost is elsewhere | ~20–50 µs (launch + Python) | +0.9 to +2.2 ms | 29.3 / 30.5 / 31.6 ms |

**Decision rule, fixed in advance:**

- slope **≥ 10 ms/K** ⇒ break count is the driver ⇒ **kernel justified**
  (removing all 43 breaks should recover a large share of 19.42 ms).
- slope **≤ 3 ms/K** ⇒ break count is NOT the driver ⇒ **do not write the
  kernel**; re-target the clone / gap-op dispatch / capture-region cost.
- between 3 and 10 ⇒ partial; recompute the expected recovery as
  `43 × measured_c_break` and judge against the engineering cost.

**My honest prior:** ~60/40 toward H-small. 452 µs per boundary is very large
for what should be an end-capture + launch + Python call, which makes me
suspect the clone (43 × a `[tokens, hidden]` copy) and per-rank convoy effects
carry more of the 19.42 ms than the segment count does. Recording this so the
result is scored either way.

**Config (matches ledger `h1-rs-tp4-c8-*` exactly, so numbers are comparable):**

```
env:   VLLM_USE_V2_MODEL_RUNNER=0 SLUICE_SLOTS=48 SLUICE_PIECEWISE=1
       SLUICE_ROUTER_SPLIT=1 SLUICE_RS_FAST_HIT=1 SLUICE_RS_NOOP=1
       SLUICE_EXTRA_BREAKS=<K>
serve: --revision 6976c7ff1b30a1b2cb7805021b8ba4684041f136 --trust-remote-code
       --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin
       --kv-cache-dtype fp8 --max-model-len 2048 --gpu-memory-utilization 0.55
       --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE
bench: random 32-in/200-out, --ignore-eos, --num-prompts 64, c=8
node:  own pod `sluice-kx` (other agent owns `sluice-h9` — do not touch)
```

**Baseline discipline:** K=0 is re-measured on MY node rather than taken from
entry [11], because cross-node absolute numbers are not comparable.
