# V4-Pro on 8×H100 — measured (2026-06-28)

**Environment.** IBM OpenShift, node `pokprod-b93r38s0`, 8× NVIDIA H100-80GB-HBM3,
~2 TB host RAM. Image `vllm/vllm-openai:v0.23.0-cu129-ubuntu2404`. Sluice installed
as a `vllm.general_plugins` entry point (no fork). Checkpoint: stock
`deepseek-ai/DeepSeek-V4-Pro` FP8 (805.4 GiB), TP=8, EP=8, `kv_cache_dtype=fp8`,
`enforce_eager`.

## H1 — capability (same node, same checkpoint)

| Engine | Result |
|---|---|
| stock vLLM (no Sluice), `gpu_mem=0.9` | **OOM at weight load** — workers TP2/TP5/TP6 `torch.OutOfMemoryError`; each GPU filled to ~79.2 GiB ("20.12 MiB free"). Peak 81059 MiB/GPU. |
| **Sluice**, `SLUICE_SLOTS=16`, `marlin`, `gpu_mem=0.45` | **Serves.** Model weights on GPU **6.54 GiB**; KV **27.17 GiB** (52,247 tokens; 102× concurrency @ 512 ctx). Output: `'The capital of France is' -> ' Paris. The capital of France is Paris. ...'`. No OOM. |

Load: weights into host RAM 267 s; full model-load 360 s. Host RAM held ~764 GiB of experts (of ~2 TB).

## V4-Pro decode throughput (Sluice, EP=8, slots=16, eager, greedy)

| batch | decode tok/s | total tok/s | prefill s |
|---|---|---|---|
| 1 | 5.69 | 5.69 | 0.18 |
| 8 | 27.81 | 26.33 | 1.32 |
| 16 | 35.28 | 33.78 | 1.74 |

**Caveats (honest framing).**
- At `slots=16` the per-step working set is exceeded at batch 16 (and during the
  startup profiling run): Sluice logs `expert_cache_slots=16 is smaller than the
  experts selected` and drops experts on those steps, so batch-16 output is **not
  guaranteed bit-exact**. For bit-exact serving, size `SLUICE_SLOTS` to the prefill
  working set or cap `--max-num-batched-tokens` (see the V2-Lite working-set data).
- Eager mode only (no CUDA graphs), single replica, no cross-layer prefetch — these
  are the documented throughput headroom levers, not measured here.
- V4-Pro has **no single-node resident baseline** (it OOMs), so these are absolute
  operating points, not a "% of resident".
