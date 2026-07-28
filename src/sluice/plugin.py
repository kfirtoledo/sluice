# SPDX-License-Identifier: Apache-2.0
"""Sluice vLLM plugin entry point.

Registered under the ``vllm.general_plugins`` entry-point group (see
``pyproject.toml``). vLLM calls :func:`register` in every engine/worker process
at startup, before the model is built. When ``SLUICE_SLOTS`` is set, we
monkeypatch vLLM's ``create_offloader`` factory so the active offloader becomes
the Sluice :class:`~sluice.offloader.ExpertStreamOffloader`. Everything else
flows through vLLM's existing ``BaseOffloader`` lifecycle (``wrap_modules`` ->
``post_init``) and the offloader's own ``quant_method.apply`` wrapping, so no
vLLM source files are modified.
"""

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

SLOTS_ENV = "SLUICE_SLOTS"

# The live offloader instance for this process (set by _create_offloader);
# lets lifecycle patches (load_model -> attach_model) reach it.
_ACTIVE = None


def register() -> None:
    """vLLM ``general_plugins`` hook. No-op unless ``SLUICE_SLOTS`` is set."""
    raw = os.environ.get(SLOTS_ENV)
    if not raw:
        return
    try:
        slots = int(raw)
        assert slots > 0
    except (ValueError, AssertionError):
        logger.warning("Sluice: ignoring invalid %s=%r (want a positive int).",
                       SLOTS_ENV, raw)
        return

    # Sluice hooks the V1 model runner's create_offloader/post_init lifecycle;
    # the V2 runner never calls it, so Sluice would silently no-op (experts
    # left resident, or OOM) AND its safety guards would never run. Refuse an
    # explicit user opt-in to V2; otherwise force V1. The force matters on
    # vLLM >= 0.25, where use_v2_model_runner defaults ON per-architecture
    # (DeepseekV2/Qwen2Moe/GraniteMoe) when the env var is unset — exactly the
    # MoE models Sluice targets. vllm.envs reads the environment lazily, and
    # register() runs in every engine/worker process before the config
    # property is consulted, so setting the env var here pins V1 everywhere.
    try:
        import vllm.envs as vllm_envs

        if getattr(vllm_envs, "VLLM_USE_V2_MODEL_RUNNER", None):
            raise RuntimeError(
                "Sluice requires the V1 model runner but VLLM_USE_V2_MODEL_RUNNER "
                "is set — the V2 runner never invokes create_offloader/post_init, "
                "so Sluice (and its correctness guards) would silently not run. "
                "Unset VLLM_USE_V2_MODEL_RUNNER."
            )
    except ImportError:
        pass
    if os.environ.get("VLLM_USE_V2_MODEL_RUNNER") is None:
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        logger.info(
            "Sluice: pinned VLLM_USE_V2_MODEL_RUNNER=0 (the V1 runner hosts "
            "the offloader lifecycle; vLLM >= 0.25 would otherwise default "
            "some MoE architectures to V2).")

    # CORRECTNESS, not tuning. Router-split emits INVALID OUTPUT when Inductor
    # compilation and CUDA-graph capture are both active (Etelis/sluice #4;
    # vLLM 0.23/0.25/0.26 alike). VLLM_USE_BREAKABLE_CUDAGRAPH=1 is what
    # decouples them: vLLM sets compilation_config.mode = NONE while keeping
    # capture, so the eager gap comes from BreakableCUDAGraphCapture.add_eager()
    # instead of the fx splitter. vLLM auto-enables it for DeepSeek-V4 and
    # MiniMax-M3-Sparse only (vllm/config/vllm.py); every other MoE model has
    # to be told, and a user who forgets gets plausible-but-wrong tokens with
    # no error. So pin it rather than document it.
    #
    # The cost is real and is accepted deliberately: this disables torch.compile
    # for the model. On DeepSeek-V4 that is free (vLLM already sets mode=NONE).
    # On smaller models it is not, but a slower correct answer beats a faster
    # wrong one. Set VLLM_USE_BREAKABLE_CUDAGRAPH=0 explicitly to opt out --
    # only do that with router-split off.
    if (os.environ.get("SLUICE_ROUTER_SPLIT", "0") == "1"
            and os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") is None):
        os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        logger.info(
            "Sluice: pinned VLLM_USE_BREAKABLE_CUDAGRAPH=1 — router-split "
            "requires it for CORRECT OUTPUT (Inductor + capture together "
            "produce invalid results). This disables torch.compile for this "
            "model; that is intended.")

    import vllm.model_executor.offloader.base as base_mod

    from sluice.offloader import ExpertStreamOffloader

    def _create_offloader(offload_config):  # noqa: ANN001
        global _ACTIVE
        _ACTIVE = ExpertStreamOffloader(expert_cache_slots=slots)
        return _ACTIVE

    # Patch the factory at its definition...
    base_mod.create_offloader = _create_offloader
    # ...and at every module that imported it by name before us.
    try:
        import vllm.v1.worker.gpu_model_runner as gmr

        if hasattr(gmr, "create_offloader"):
            gmr.create_offloader = _create_offloader
        # attach_model needs the model handle: it arms ROUTER-SPLIT (patches
        # each MoE runner's _forward_entry when SLUICE_ROUTER_SPLIT=1) and,
        # under SLUICE_LAZY_STEP, wraps model.forward for the one-boundary
        # miss check. load_model is the earliest point self.model exists.
        _orig_load = gmr.GPUModelRunner.load_model

        def load_model(self, *a, **k):  # noqa: ANN001
            out = _orig_load(self, *a, **k)
            try:
                mdl = getattr(self, "model", None)
                if _ACTIVE is not None and mdl is not None:
                    _ACTIVE.attach_model(mdl)
                # TELL vLLM ABOUT THE SLOT CACHE.
                # vLLM captures ``model_memory_usage`` inside the
                # DeviceMemoryProfiler that wraps model loading
                # (gpu_model_runner.py:5294) and only afterwards calls
                # ``get_offloader().post_init()`` (:5381), which is where the
                # slot buffers are allocated. So the slot cache misses
                # ``weights_memory``; and because it is allocated BEFORE the
                # profile run it also sits in the baseline and never shows up
                # in ``torch_peak_increase``. Both accounting paths miss it.
                #
                # Measured consequence (Qwen3-30B-A3B, TP=2, default util):
                # vLLM reported "Available KV cache memory: 70.29 GiB" at
                # slots=64 and 70.58 GiB at slots=96 -- a 6.5 GiB difference in
                # live buffers moving the estimate by 0.29 GiB -- then died in
                # gpu_worker.initialize_from_config allocating a KV cache that
                # could not fit. This is why every Sluice benchmark has needed
                # a hand-lowered --gpu-memory-utilization, with no error
                # message ever naming Sluice.
                #
                # This wrapper runs after post_init and before
                # determine_available_memory, so it is the right seam to
                # correct the figure.
                sb = int(getattr(_ACTIVE, "slot_vram_bytes", 0) or 0)
                if sb and hasattr(self, "model_memory_usage"):
                    self.model_memory_usage += sb
                    logger.info(
                        "Sluice: reported %.2f GiB of GPU slot cache to vLLM's "
                        "memory profiler (model_memory_usage %.2f -> %.2f GiB) "
                        "so the KV cache is sized against real free memory.",
                        sb / (1 << 30),
                        (self.model_memory_usage - sb) / (1 << 30),
                        self.model_memory_usage / (1 << 30),
                    )
            except Exception:
                logger.exception(
                    "Sluice: attach_model failed (router-split/lazy-step "
                    "not armed).")
            return out

        gmr.GPUModelRunner.load_model = load_model
    except Exception:  # pragma: no cover - defensive across vLLM versions
        logger.debug("Sluice: gpu_model_runner not patchable; relying on base.")

    logger.info("Sluice: activated via %s=%d (expert offloading on).",
                SLOTS_ENV, slots)
