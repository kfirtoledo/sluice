# Benchmark pods (H100, OpenShift)

Throwaway pods used to benchmark Sluice. **Not a production deployment** — no
Deployment, no Service, nothing served outside the pod. Each is a
`vllm/vllm-openai` container running `sleep infinity` that you `kubectl exec`
into.

| file | GPUs | model |
|---|---|---|
| `pod-v2.yaml` | 1 | DeepSeek-V2-Lite (TP=1) |
| `pod-qw2.yaml` | 2 | Qwen3-30B-A3B (TP=2) |
| `pod-kx.yaml` | 4 | DeepSeek-V4-Flash (TP=4 + EP) |

```bash
kubectl apply -f examples/k8s/pod-qw2.yaml
kubectl exec -n storage sluice-qw2 -- bash -lc '
  export HOME=/work USER=sluice LOGNAME=sluice   # OpenShift random UID
  cd /work/sluice && pip install -e . --no-deps --no-build-isolation'
```

## Sizing the launch flags

Two settings are **not** free choices with `SLUICE_ROUTER_SPLIT=1`.

**1. `--max-num-batched-tokens ≤ SLUICE_SLOTS // top_k`** — enforced; Sluice
raises at startup otherwise. Router-split is *single-wave*: a step of `T`
tokens routes up to `T × top_k` distinct experts, and all of them must be
resident at once, so `T ≤ slots / top_k`.

| model | top_k | slots | cap |
|---|---|---|---|
| Qwen3-30B-A3B | 8 | 64 / 96 | 8 / 12 |
| DeepSeek-V2-Lite | 6 | 48 | 8 |
| DeepSeek-V4-Flash | 6 | 48 | 8 |

vLLM's default is **8192**, so this is a large restriction: in decode each
sequence contributes one token, so a cap of 8 means **at most 8 sequences
decode per step**. Concurrency beyond that queues rather than batches.
The **classic path (no `SLUICE_ROUTER_SPLIT`) has no envelope** and runs at the
default.

**2. `-cc.cudagraph_mode`** — `PIECEWISE` is required on torch.compile models
(Qwen3, V2-Lite): the eager gap exists *because* Sluice adds `moe_forward` to
`splitting_ops`, so a FULL graph would capture the hook's D2H sync. Sluice
refuses it. On models vLLM runs uncompiled via breakable cudagraph
(DeepSeek-V4), `SLUICE_ALLOW_FULL_CG=1` keeps vLLM's FULL decode graphs and is
worth ~28 % TPOT.

`--gpu-memory-utilization` needs **no** override: Sluice reports its slot cache
to vLLM's memory profiler, so the KV cache is sized against real free memory.

## Launch commands

The pod spec carries **no** Sluice or vLLM flags — the container runs
`sleep infinity`. Everything below goes on the `vllm serve` line inside it.
`SLUICE_*` variables are read in `src/sluice/offloader.py`.

Common to every model:

```bash
export HOME=/work USER=sluice LOGNAME=sluice   # OpenShift random UID
export HF_HOME=/vllm-cache/hf
export SLUICE_ROUTER_SPLIT=1 SLUICE_PIECEWISE=1 SLUICE_RS_FAST_HIT=1
```

### Qwen3-30B-A3B — `pod-qw2.yaml`, 2 GPU

```bash
export SLUICE_SLOTS=96                      # 96/128 experts resident
vllm serve Qwen/Qwen3-30B-A3B --trust-remote-code \
  --tensor-parallel-size 2 --moe-backend triton --max-model-len 2048 \
  --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE
```

`slots=96` allows a cap of 12 (`96 // top_k 8`), but **8 measured faster at
c=6** (13.21 vs 14.78 ms): a larger budget lets prefill chunks share decode
steps and raises TPOT. Use the cap as a ceiling, not a target.

### DeepSeek-V2-Lite — `pod-v2.yaml`, 1 GPU

```bash
export SLUICE_SLOTS=48                      # 48/64 experts resident
vllm serve deepseek-ai/DeepSeek-V2-Lite --trust-remote-code \
  --tensor-parallel-size 1 --moe-backend triton --max-model-len 2048 \
  --max-num-batched-tokens 8 -cc.cudagraph_mode=PIECEWISE
```

### DeepSeek-V4-Flash — `pod-kx.yaml`, 4 GPU

```bash
export SLUICE_SLOTS=48 SLUICE_ALLOW_FULL_CG=1
vllm serve deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
  --revision 6976c7ff1b30a1b2cb7805021b8ba4684041f136 \
  --tensor-parallel-size 4 --enable-expert-parallel --moe-backend marlin \
  --kv-cache-dtype fp8 --max-model-len 2048 --max-num-batched-tokens 8
```

**No `-cc.cudagraph_mode` here.** V4 runs on vLLM's breakable-cudagraph path,
where the eager gap comes from `add_eager` rather than `splitting_ops`, so
`SLUICE_ALLOW_FULL_CG=1` keeps vLLM's FULL decode graphs — worth **−28 % TPOT
and +38 % throughput** (49.52 → 35.66 ms, 132 → 182 tok/s). On the
torch.compile models above, Sluice refuses FULL and `PIECEWISE` is required.

### Flags at a glance

| flag | required? | why |
|---|---|---|
| `SLUICE_SLOTS` | yes | experts kept on GPU per layer; the rest stream from host RAM |
| `SLUICE_ROUTER_SPLIT=1` | fast path | splits the MoE layer so all but the gap is captured |
| `SLUICE_PIECEWISE=1` | with router-split | permits non-eager execution |
| `--max-num-batched-tokens` | **enforced** | `≤ SLUICE_SLOTS // top_k`; startup fails otherwise |
| `-cc.cudagraph_mode=PIECEWISE` | torch.compile models | the gap exists only via `splitting_ops` |
| `SLUICE_ALLOW_FULL_CG=1` | breakable-cudagraph models | keeps FULL decode graphs (V4: −28 %) |
| `--moe-backend` | yes | must honor `expert_map`: `marlin` (FP8/NVFP4) or `triton` (unquantized) |
| `--gpu-memory-utilization` | **no** | Sluice reports its slot cache to vLLM's profiler; leave at default |

## Cluster notes

* `/work` is an `emptyDir` — anything written there dies with the pod.
* Export `HOME`/`USER`/`LOGNAME` or pip and torch break under a random UID.
* `pkill -f 'vllm serve'` matches your own `kubectl exec`. Use
  `pkill -f 'bin/vllm'`.
* After killing a run, check `nvidia-smi --query-compute-apps` for orphaned
  `VLLM::Worker_TP` processes — they survive and hold ~62 GiB/GPU.
* `nodeAffinity` excludes a node whose broken NVLink fails pod admission
  (`failed to get nvlink state: GPU requires reset`). Adjust for your cluster.
