# July 2026 validation campaign — benchmark index

`campaign_202607.csv` holds every `vllm bench serve` run from the 2026-07-03…05
cluster campaign (pokprod001, H100 nodes, vLLM v0.23.0). One row per bench;
experiment tags map to:

| experiment | what it tested |
|---|---|
| `results_nonmla` / `results_nonmla2` | non-MLA exact decode classifier A/B on Qwen1.5-MoE (negative: heuristic wins) + protect_frac regime test |
| `results_pfsweep` | protect_frac sweep + resident reference on Qwen (noise; not a lever) |
| `results_refix` | V4-Pro EP=8 serving re-measure post red-team fixes + short-prompt coverage |
| `results_mtp` | MTP k=1 on random-token prompts (acceptance floor; break-even analysis) + same-day plain baselines |
| `results_mtp2` | MTP k=1 on ShareGPT (the +67% single-stream result) |
| `results_mtp3` | MTP k=2 (user run + matched-np confirmation) + tighter plain c1 |
| `results_abattr` | interleaved ABAB fixed-vs-pre-fix offloader (fix costs nothing; ±15% c16 day spread) |
| `results_r6` | NUMA H2D probe (~1.3%, not a lever) + slots-vs-KV sweep on V4-Pro (s16/s22/s25: +20%) |
| `results_dpv4_4g` | TP=4/EP=4 reference + 16k-context sizing-rule check (+9–20%) |
| `results_glmserve` | GLM-4.5-Air resident + s12@0.50 vs s20@0.35 (sizing rule 2.7× on bf16) |
| `results_glm51` | GLM-5.1-FP8 capability pair: stock OOM vs Sluice serves (floor config) |
| `results_glm51mtp` | GLM-5.1 tuned config + MTP k=1 (+58% single-stream; physics replicates) |
| `results_dp` / `results_dptp2` / `results_race*` / `results_stress` | DP correctness matrix, race elimination and concurrency stress (status files only; bit-compare verdicts in docs/EVALUATION.md) |

Acceptance-rate excerpts, DIAG hit-rate lines, and status trails for every run
are in the same directories on the `sluice-work` PVC; the curated conclusions
live in `docs/EVALUATION.md`.
