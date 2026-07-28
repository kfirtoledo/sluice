# Benchmark harness — pods and run scripts

**Not part of the Sluice product.** These are the throwaway pods and scripts the
performance campaign runs on; `master` carries no Kubernetes manifests. They are
tracked so results are reproducible, not because Sluice ships a deployment.

## Pods

| file | GPUs | used for |
|---|---|---|
| `pod-v2.yaml` | 1 | DeepSeek-V2-Lite (TP=1) |
| `pod-kx.yaml` | 4 | DeepSeek-V4-Flash (TP=4 + EP) |
| `pod-qw2.yaml` | 2 | Qwen3-30B-A3B (TP=2) |

All three are the same shape: a `vllm/vllm-openai:v0.25.1` container running
`sleep infinity`, that benchmarks are `kubectl exec`'d into. There is no
Deployment and no Service — nothing is served outside the pod.

Volumes: the `vllm-cache` PVC holds the HF model cache (mounted `/vllm-cache`,
`HF_HOME=/vllm-cache/hf`); `/dev/shm` is a memory emptyDir for NCCL; `/work` is
an emptyDir, so **everything written there is lost when the pod dies** — keep
results in git.

`nodeAffinity` excludes `pokprod-b93r44s0`, which fails pod admission with
`failed to get nvlink state: GPU requires reset`. Without that exclusion the
scheduler keeps landing pods there and they fail before starting, which looks
like the namespace being wiped.

## Running

```bash
kubectl apply -f scripts/pod-qw2.yaml
# install Sluice into the pod (editable, no deps — vLLM is already in the image)
kubectl cp <worktree>/src ... ; pip install -e . --no-deps --no-build-isolation
kubectl cp scripts/<run>.sh storage/<pod>:/work/run.sh
kubectl exec -n storage <pod> -- bash -lc \
  'export HOME=/work USER=sluice LOGNAME=sluice; setsid nohup bash /work/run.sh &'
```

`HOME`/`USER`/`LOGNAME` are required: OpenShift assigns a random UID and both
pip and torch break without them.

`pkill -f 'vllm serve'` kills your own `kubectl exec` (its command line matches).
Use `pkill -f 'bin/vllm'` or a bracket pattern. After killing a run, check for
orphaned `VLLM::Worker_TP` processes — they survive and hold ~62 GiB/GPU.
