# SPDX-License-Identifier: Apache-2.0
"""Routing-aware MoE expert-weight streaming offloader (Sluice).

Keeps a FusedMoE layer's per-rank expert shard in host CPU memory and streams
only the experts the router selects into a small fixed-size GPU cache each
forward pass. The resident working set follows the router's top-k decisions
rather than being fixed at load time.

Load path (works for unquantized and quantized experts):
  1. ``wrap_modules`` (model construction, before checkpoint load): move each
     MoE layer's expert params to CPU so the loader fills host memory, and
     record the layer for later cache installation.
  2. The loader runs ``process_weights_after_loading`` inside vLLM's
     ``device_loading_context``, which stages each layer's experts back onto the
     GPU for processing (e.g. a CUDA-only FP4/FP8 marlin repack) and then
     restores them to CPU. Only one layer's experts are on the GPU at a time,
     so an expert shard far larger than GPU memory can still be processed.
  3. ``post_init`` (after load completes, i.e. after the context restore):
     install a small ``[num_slots, ...]`` GPU cache per layer as the live
     weights, hold the processed experts in host memory, and wrap the layer's
     ``quant_method.apply`` so the routing hook fires every forward.

Installing the cache in ``post_init`` (not inside ``process_weights``) is
required: ``device_loading_context`` records each param's pre-context device
(CPU, because ``wrap_modules`` parked it there) and restores it on exit, which
would move our GPU cache straight back to CPU.

The kernel reads ``layer.expert_map`` (global expert id -> resident row) to
index expert weights; this offloader rewrites that map per step to point the
selected experts at their GPU cache slots, so no MoE kernel changes are needed.
Requires a backend that applies ``expert_map`` (TRITON for unquantized, MARLIN
for NVFP4/FP8); FlashInfer-style monolithic backends ignore it.

Execution model (per MoE layer, per step)
-----------------------------------------
A single fused-MoE kernel launch reads one expert map, so every expert it is
to compute must be resident simultaneously. Instead of requiring
``slots >= per-step working set`` (which a prefill chunk pushes toward *all*
experts), the offloader partitions the step's selected experts into **waves**
that each fit the cache and calls the wrapped kernel once per wave with a map
exposing only that wave, summing the partial outputs. ``topk_weights`` are
computed before ``apply`` and an unmapped expert contributes zero — exactly
how expert-parallel ranks skip remote experts — so the wave sum equals a
single full launch (partials are summed in fp32 to keep the drift at ulp
level; a single-wave step is bit-identical).

Cache policy is **SLRU** (scan-resistant): a *protected* segment holds experts
re-referenced by decode steps, a *probationary* segment absorbs one-shot
streams. The step's decode-selected experts (identified from the v1 forward
context) may promote into and evict from the protected segment; prefill/scan
experts are confined to probation, so a chunked-prefill scan can never flush
the decode-hot set. Multi-wave steps rotate through probation in ping-pong
slot groups with wave k+1's misses prefetched on a dedicated copy stream while
wave k computes.
"""

import itertools
import os
from collections import OrderedDict
from collections.abc import Generator
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.model_executor.offloader.base import BaseOffloader, should_pin_memory

logger = init_logger(__name__)

# Shared/global tensors (broadcast activation scales, bias/index tables), not
# per-expert weights. Mirrors the exclusions RoutedExperts applies for EPLB.
NON_EXPERT_WEIGHTS = frozenset(
    {
        "e_score_correction_bias",
        "w13_input_scale",
        "w2_input_scale",
        "hash_indices_table",
    }
)

PROTECT_FRAC_ENV = "SLUICE_PROTECT_FRAC"
STATS_EVERY_ENV = "SLUICE_STATS_EVERY"
LFU_ENV = "SLUICE_LFU"
# Halve the per-expert popularity counts every this many real steps, so the
# protected set tracks the *current* hot experts instead of pinning whatever
# was hot at startup (frequency aging — prevents stale LFU pinning).
LFU_AGE_STEPS = 256


@dataclass
class _ExpertLayerCache:
    """Per-layer streaming state for a single ``RoutedExperts`` module."""

    local_num_experts: int
    num_slots: int
    protected_cap: int
    device: torch.device
    param_names: list[str] = field(default_factory=list)
    gpu_cache: dict[str, torch.Tensor] = field(default_factory=dict)
    cpu_store: dict[str, torch.Tensor] = field(default_factory=dict)
    local_of: list[int] = field(default_factory=list)
    expert_map_buf: torch.Tensor | None = None
    map_host: torch.Tensor | None = None
    slot_of: dict[int, int] = field(default_factory=dict)
    expert_in_slot: list[int | None] = field(default_factory=list)
    free_slots: list[int] = field(default_factory=list)
    # SLRU segments: slot -> None, insertion order = recency (last = MRU).
    protected: OrderedDict = field(default_factory=OrderedDict)
    probation: OrderedDict = field(default_factory=OrderedDict)
    bytes_per_expert: int = 0
    # LFU hybrid (opt-in): decayed popularity count per local expert, and a
    # per-layer real-step counter driving periodic aging.
    freq: dict = field(default_factory=dict)
    lfu_steps: int = 0
    # Full residency: the slot cache holds every local expert, so the expert
    # map is the identity local_of and never changes — the routing hook is
    # pure overhead and is skipped entirely (behaves like a resident EP rank).
    static_full: bool = False
    # local slot index -> global expert id (inverse of local_of); used by the
    # contiguous prefill block path.
    global_of_local: list = field(default_factory=list)
    # Recorded on the compute stream right after the async pinned map H2D;
    # synchronized before the next mutation of map_host so a still-in-flight
    # copy is never torn. (Today the per-layer host sync already drains it;
    # the event keeps that safety explicit and survives removing the sync.)
    map_ev: "torch.cuda.Event | None" = None


class _StepStats:
    """Cheap counters, bucketed by step class ('small' ~ decode-like,
    'scan' ~ prefill-like working sets)."""

    __slots__ = ("steps", "hits", "misses", "bytes", "waves")

    def __init__(self) -> None:
        self.steps = 0
        self.hits = 0
        self.misses = 0
        self.bytes = 0
        self.waves = 0


class ExpertStreamOffloader(BaseOffloader):
    """Stream routed-expert weights CPU->GPU per forward pass, by routing.

    Args:
        expert_cache_slots: GPU-resident expert slots per MoE layer, per rank.
            Any positive count is *correct* (a step selecting more experts than
            slots executes in multiple waves); larger counts are faster.
    """

    def __init__(self, expert_cache_slots: int):
        assert expert_cache_slots > 0
        self.expert_cache_slots = expert_cache_slots
        # Speculative-decode draft depth (0 = none). Under spec-decode/MTP a
        # genuine decode request verifies (1 + k) tokens, so the decode region
        # is ndec*(1+k) rows, not ndec — the pure-decode gate accounts for it.
        # Populated from the vLLM config in _check_config.
        self._spec_tokens = 0
        # DP>1 (SLUICE_ALLOW_DP=1): routing is valid (naive dispatch gathers
        # logits before select), but per-rank row-layout signals are not —
        # set in _check_config; disables all promotion paths in the hook.
        self._dp_mode = False
        # Per-layer wave plans for the modular-kernel (MK) path, keyed by
        # id(module): built in the wrapped mk._prepare from POST-dispatch
        # topk_ids, consumed by the wrapped mk._fused_experts.
        self._mk_waves: dict[int, list] = {}
        # Experts-kernel class name per wrapped layer — keys the wave
        # allowlist (map-driven kernels wave; count-driven kernels fault on
        # a wave map that hides a selected expert).
        self._mk_experts_name: dict[int, str] = {}
        # MK-path fills default to the COMPUTE stream (no cross-stream
        # edges): a stream race in the overlapped path crashes marlin-timing
        # kernels (CUDA_LAUNCH_BLOCKING=1 makes the same config pass, so the
        # fault is ordering, not semantics). SLUICE_MK_SYNC_FILLS=0 restores
        # copy-stream overlap for experimentation.
        self._mk_sync_fills = (
            os.environ.get("SLUICE_MK_SYNC_FILLS", "1") != "0"
        )
        self.pin_memory = should_pin_memory()
        if not self.pin_memory:
            # Without pinned host memory, ``non_blocking=True`` H2D copies
            # silently fall back to synchronous staged transfers — the whole
            # streaming path serializes behind compute (a large, invisible
            # slowdown). Surface it loudly rather than let it be mistaken for
            # streaming overhead.
            logger.warning(
                "Sluice: pinned host memory is unavailable — expert streaming "
                "will use synchronous H2D copies and be substantially slower. "
                "Ensure the host has enough lockable RAM for the expert store."
            )
        self._caches: dict[int, _ExpertLayerCache] = {}
        self._layers: list[nn.Module] = []
        try:
            frac = float(os.environ.get(PROTECT_FRAC_ENV, "0.5"))
        except ValueError:
            frac = 0.5
        self.protect_frac = min(max(frac, 0.0), 0.9)
        # LFU hybrid (opt-in): make the protected segment a popularity-priority
        # set (evict least-frequently-used) instead of pure LRU. Targets
        # covered-but-churning decode cells where the working set fits but the
        # *identity* of the hot experts drifts step-to-step. Probation stays
        # LRU (it is the scan buffer, where recency is the right signal).
        self.lfu = os.environ.get(LFU_ENV, "0") == "1"
        # Contiguous block streaming for pure-prefill steps (opt-in). Measured
        # neutral on V2-Lite (its per-expert copies are already few and large);
        # the potential win is the many-small-copy FP8 marlin format (V4-Pro),
        # unvalidated while the 8-GPU node is unavailable. Off until proven.
        self.prefill_blocks = os.environ.get("SLUICE_PREFILL_BLOCKS", "0") == "1"
        # Exact decode classifier for non-MLA backends (query_start_loc
        # based). OPT-IN: measured on Qwen1.5-MoE (balanced aux-loss-free
        # routing, 1xH100, slots=30) it loses 10-16% tok/s to the ws
        # heuristic at c2-c4 and is neutral at c16-32 — with no popularity
        # skew there is no stable decode-hot set, so exact decode
        # identification only adds promotion churn while pinning slots the
        # prefill scan needs. Enable for models with real expert skew.
        self.nonmla_classifier = (
            os.environ.get("SLUICE_NONMLA_CLASSIFIER", "0") == "1"
        )
        self._copy_stream: torch.cuda.Stream | None = None
        self._warned_waves = False
        # Diagnostics (SLUICE_DIAG=1): per-step working-set high-water marks +
        # periodic hit-rate summaries (every SLUICE_STATS_EVERY hook calls).
        self._diag = bool(os.environ.get("SLUICE_DIAG"))
        try:
            self._stats_every = int(os.environ.get(STATS_EVERY_ENV, "500"))
        except ValueError:
            self._stats_every = 500
        self._ws_hwm: dict[int, int] = {}
        self._stats = {"small": _StepStats(), "scan": _StepStats()}
        self._hook_calls = 0
        logger.info(
            "Sluice ExpertStreamOffloader enabled (%d cache slots per layer, "
            "protect_frac=%.2f).",
            expert_cache_slots,
            self.protect_frac,
        )

    @staticmethod
    def _is_moe_layer(module: nn.Module) -> bool:
        return (
            hasattr(module, "local_num_experts")
            and hasattr(module, "global_num_experts")
            and hasattr(module, "quant_method")
        )

    @staticmethod
    def _per_expert_params(module: nn.Module, local_n: int) -> dict[str, nn.Parameter]:
        return {
            name: p
            for name, p in module.named_parameters(recurse=False)
            if name not in NON_EXPERT_WEIGHTS and p.dim() >= 1 and p.shape[0] == local_n
        }

    # -- model construction (pre-load) --------------------------------------

    def wrap_modules(
        self,
        modules_generator: Generator[nn.Module, None, None],
    ) -> list[nn.Module]:
        # Consume lazily; prepare each layer as it is built so the GPU never
        # holds more than one layer's experts at a time during load.
        modules = []
        for module in modules_generator:
            for sub in module.modules():
                if self._is_moe_layer(sub):
                    self._prepare_layer(sub)
            modules.append(module)
        return modules

    def _prepare_layer(self, module: nn.Module) -> None:
        """Move a layer's (still-empty) experts to CPU so the checkpoint loads
        into host memory, and record the layer for cache installation."""
        quant_method = module.quant_method
        if quant_method is None or getattr(quant_method, "is_monolithic", False):
            return
        if getattr(module, "rocm_aiter_fmoe_enabled", False):
            return
        local_n = int(module.local_num_experts)
        per_expert = self._per_expert_params(module, local_n)
        if not per_expert:
            return

        for p in per_expert.values():
            p.data = p.data.to("cpu")

        if all(id(module) != id(m) for m in self._layers):
            self._layers.append(module)

    # -- after load (post_init, after device_loading_context restore) -------

    def post_init(self) -> None:
        self._check_config()
        if torch.cuda.is_available():
            # The loader's device_loading_context restore may still have
            # copies in flight; cpu_store aliases those host buffers, so make
            # sure they are fully materialized before we pin/serve them.
            torch.cuda.synchronize()
        for module in self._layers:
            self._install_cache(module)
            self._wrap_apply(module)
            torch.accelerator.empty_cache()
        # Operator visibility: how much host memory the expert store holds and
        # how much VRAM the slot caches cost (per rank). The host figure is the
        # amount pinned when pin_memory is on — the value to size lockable RAM
        # against on large models.
        if self._caches:
            # static_full layers keep no host copy (resident-rank); only
            # streaming layers hold a (pinned) host expert store.
            host_bytes = sum(
                c.bytes_per_expert * c.local_num_experts
                for c in self._caches.values()
                if not c.static_full
            )
            vram_bytes = sum(
                c.bytes_per_expert * c.num_slots for c in self._caches.values()
            )
            logger.info(
                "Sluice: %d MoE layers ready — host expert store %.1f GiB%s, "
                "GPU slot cache %.1f GiB (per rank).",
                len(self._caches),
                host_bytes / (1 << 30),
                " (pinned)" if self.pin_memory else " (pageable)",
                vram_bytes / (1 << 30),
            )

    def _check_config(self) -> None:
        """Fail fast on configurations that would corrupt outputs silently,
        and record the speculative-decode draft depth.

        - DP>1: naive DP gathers hidden states AND topk_ids from all DP ranks
          *inside* ``apply``, so the hook's pre-dispatch topk_ids no longer
          describe the processed tokens — wrong experts, no error.
        - EPLB: re-registers the layer's expert-map buffer over Sluice's.
        - CUDA graphs: capture would freeze one step's expert map into the
          graph; every replay computes with stale routing.
        """
        # Fail CLOSED: if the config can't be read, we can't verify any of the
        # silent-corruption guards below, so refuse rather than serve blind.
        try:
            from vllm.config import get_current_vllm_config

            cfg = get_current_vllm_config()
        except Exception as e:
            raise RuntimeError(
                "Sluice: could not read the vLLM config to verify it is safe "
                "(DP/EPLB/eager); refusing to run. This usually means an "
                "unexpected vLLM version."
            ) from e
        if cfg is None:
            raise RuntimeError(
                "Sluice: vLLM config is None; cannot verify a safe "
                "configuration (DP/EPLB/eager). Refusing to run."
            )
        pc = getattr(cfg, "parallel_config", None)
        if pc is not None:
            if getattr(pc, "data_parallel_size", 1) > 1:
                # Naive DP dispatch (non-modular quant methods: triton/marlin)
                # all-gathers hidden states AND router logits across DP ranks
                # BEFORE select_experts/apply, so the hook sees post-gather
                # routing and streaming is correct. What is NOT valid under DP
                # is the per-rank row-layout assumptions of the decode
                # classifiers (gathered rows interleave ranks), so promotion
                # is force-disabled (_dp_mode) — SLRU degrades to a
                # scan-resistant LRU. Modular-kernel quant methods
                # (supports_internal_mk: DeepEP-style dispatch inside apply)
                # remain unsafe and are rejected per layer in _install_cache.
                if os.environ.get("SLUICE_ALLOW_DP", "0") != "1":
                    raise RuntimeError(
                        "Sluice: DP>1 support is experimental — the decode "
                        "classifier is disabled under DP (gathered rows "
                        "interleave ranks) and only naive-dispatch backends "
                        "(triton/marlin) are safe. Set SLUICE_ALLOW_DP=1 to "
                        "enable."
                    )
                self._dp_mode = True
                logger.warning(
                    "Sluice: DP>1 enabled (SLUICE_ALLOW_DP=1) — decode "
                    "promotion disabled; cache policy is scan-resistant LRU."
                )
                # Custom fusion passes race with wave-looped MoE under DP
                # (measured: CUDA illegal access on V4-Pro DP=2xTP=2; clean
                # with them disabled). Warn unless they are explicitly off —
                # defaults resolve per-model, so we cannot verify from here.
                pcfg = getattr(
                    getattr(cfg, "compilation_config", None), "pass_config", None
                )
                fusions = ("fuse_allreduce_rms", "fuse_norm_quant",
                           "fuse_act_quant")
                if pcfg is None or any(
                    getattr(pcfg, k, None) is not False for k in fusions
                ):
                    logger.warning(
                        "Sluice: DP>1 with custom fusion passes not "
                        "explicitly disabled — fused collectives race with "
                        "wave-looped MoE (CUDA illegal access). Run with "
                        "--compilation-config '{\"pass_config\":"
                        "{\"fuse_allreduce_rms\":false,\"fuse_norm_quant\":"
                        "false,\"fuse_act_quant\":false}}' and "
                        "VLLM_ALLREDUCE_USE_FLASHINFER=0."
                    )
            if getattr(pc, "enable_eplb", False):
                raise RuntimeError(
                    "Sluice: EPLB is unsupported (it re-registers the expert "
                    "map buffer over Sluice's slot map)."
                )
        # Require eager. enforce_eager guarantees the MoE region is not
        # captured/compiled: a CUDA graph would bake one step's expert map into
        # the replay (stale routing → silently wrong), and stock torch.compile
        # (mode != NONE) skips the post_init where Sluice installs the cache
        # (experts stranded on CPU). Both are covered by requiring eager.
        mc = getattr(cfg, "model_config", None)
        if mc is not None and not getattr(mc, "enforce_eager", False):
            raise RuntimeError(
                "Sluice: requires eager execution — run with --enforce-eager. "
                "Without it, CUDA-graph capture freezes a stale expert map "
                "(silently wrong output) and stock torch.compile skips the "
                "post_init that installs Sluice's expert cache."
            )
        cc = getattr(cfg, "compilation_config", None)
        cg = getattr(cc, "cudagraph_mode", None) if cc is not None else None
        if cg is not None and getattr(cg, "name", str(cg)) != "NONE":
            raise RuntimeError(
                "Sluice: CUDA graphs would capture a frozen expert map and "
                "replay it with stale routing (silently wrong outputs). Run "
                "with enforce_eager (--enforce-eager)."
            )
        # Speculative decode / MTP: a decode request verifies (1 + k) tokens,
        # so the decode region is ndec*(1+k) rows. Record k so the pure-decode
        # classifier doesn't mistake spec-decode steps for mixed prefill.
        sc = getattr(cfg, "speculative_config", None)
        k = getattr(sc, "num_speculative_tokens", 0) if sc is not None else 0
        try:
            self._spec_tokens = max(int(k or 0), 0)
        except (TypeError, ValueError):
            self._spec_tokens = 0

    @staticmethod
    def _infer_device(module: nn.Module) -> torch.device:
        for t in itertools.chain(
            module.parameters(recurse=False), module.buffers(recurse=False)
        ):
            if t is not None and t.device.type != "cpu":
                return t.device
        from vllm.platforms import current_platform

        idx = torch.cuda.current_device() if torch.cuda.is_available() else 0
        return torch.device(current_platform.device_type, idx)

    @staticmethod
    def _backend_ignores_expert_map(module: nn.Module) -> bool:
        """True if the layer's resolved MoE experts backend declares it does
        NOT apply ``expert_map`` (FlashInfer b12x, trtllm_mxint4, cpu_int4).
        Sluice's whole mechanism is rewriting that map, so such a backend would
        run the kernel against the truncated slot cache while ignoring the map
        — silently wrong output or an illegal expert index. Only returns True
        on a *definitive* False, so map-applying backends (triton/marlin, which
        the eval uses) are never mis-flagged."""
        qm = getattr(module, "quant_method", None)
        if qm is None:
            return False
        seen = []
        for obj in (qm, getattr(qm, "fused_experts", None)):
            if obj is None:
                continue
            seen.append(obj)
            inner = getattr(obj, "fused_experts", None)
            if inner is not None:
                seen.append(inner)
        for c in seen:
            sem = getattr(c, "supports_expert_map", None)
            if callable(sem):
                try:
                    if sem() is False:
                        return True
                except Exception:
                    pass
        return False

    def _install_cache(self, module: nn.Module) -> None:
        """Capture the layer's processed (host-resident) experts and install a
        small GPU slot cache as the live weights."""
        local_n = int(module.local_num_experts)
        global_n = int(module.global_num_experts)
        per_expert = self._per_expert_params(module, local_n)
        if not per_expert:
            return
        qm_ = getattr(module, "quant_method", None)
        if (
            self._dp_mode
            and (
                getattr(qm_, "supports_internal_mk", False)
                or type(qm_).__name__ == "FusedMoEModularMethod"
            )
            and self._find_mk(module) is None
        ):
            # An MK-owning method whose kernel object we cannot locate: the
            # dispatch happens inside apply, after the pre-dispatch hook reads
            # routing, and we have no seam to hook — refuse rather than serve
            # silently wrong output. (When the kernel IS found, _wrap_apply
            # hooks its post-dispatch seam instead — see _wrap_mk.)
            probe = {
                a: type(getattr(qm_, a, None)).__name__
                for a in ("moe_kernel", "fused_experts", "kernel")
            }
            raise RuntimeError(
                "Sluice: DP>1 with a modular-kernel quant method whose "
                "kernel object could not be located — no safe hook seam. "
                f"quant_method={type(qm_).__name__}, candidates={probe}. "
                "Drop DP for this model/vLLM combination."
            )
        if self._backend_ignores_expert_map(module):
            raise RuntimeError(
                "Sluice: the active MoE experts backend does not apply "
                "expert_map, so streaming would run the kernel against a "
                "truncated cache while ignoring Sluice's slot map (wrong "
                "output or illegal expert index). Force an expert_map backend: "
                "--moe-backend marlin (FP8/NVFP4) or triton (unquantized)."
            )
        device = self._infer_device(module)
        num_slots = min(self.expert_cache_slots, local_n)
        protected_cap = min(int(num_slots * self.protect_frac), num_slots - 1)

        cache = _ExpertLayerCache(
            local_num_experts=local_n,
            num_slots=num_slots,
            protected_cap=max(protected_cap, 0),
            device=device,
            param_names=list(per_expert.keys()),
            expert_in_slot=[None] * num_slots,
            free_slots=list(range(num_slots - 1, -1, -1)),
        )
        cache.static_full = num_slots == local_n
        for name, p in per_expert.items():
            cpu = p.data.to("cpu")
            slot = torch.empty((num_slots, *p.shape[1:]), dtype=p.dtype, device=device)
            if cache.static_full:
                # Resident-rank layer: fill every slot once from a pageable
                # host tensor, then drop it — the GPU slot is the live param
                # and nothing ever streams, so keep no (pinned) host copy.
                slot.copy_(cpu)
            else:
                if self.pin_memory:
                    cpu = cpu.pin_memory()
                cache.cpu_store[name] = cpu
            cache.gpu_cache[name] = slot
            p.data = slot
            cache.bytes_per_expert += cpu[0].nbytes

        orig_map = module.expert_map
        if orig_map is not None:
            cache.local_of = orig_map.to("cpu").tolist()
            map_dtype = orig_map.dtype
        else:
            cache.local_of = list(range(global_n))
            map_dtype = torch.int32
        # Inverse map (local slot index -> global id), for the contiguous
        # prefill block path which streams the shard in local order.
        cache.global_of_local = [-1] * local_n
        for g, local in enumerate(cache.local_of):
            if 0 <= local < local_n:
                cache.global_of_local[local] = g
        cache.expert_map_buf = torch.full(
            (global_n,), -1, dtype=map_dtype, device=device
        )
        if cache.static_full:
            # Persistent identity map (global -> its resident local slot); the
            # hook never runs, so this is the layer's only map write.
            for g, local in enumerate(cache.local_of):
                if local >= 0:
                    cache.expert_map_buf[g] = local
            for local in range(local_n):
                cache.slot_of[local] = local
                cache.expert_in_slot[local] = local
        # layer.expert_map is a property returning the _expert_map buffer.
        module.register_buffer("_expert_map", cache.expert_map_buf, persistent=False)
        map_host = torch.full((global_n,), -1, dtype=map_dtype, device="cpu")
        if self.pin_memory:
            map_host = map_host.pin_memory()
        cache.map_host = map_host

        self._caches[id(module)] = cache
        logger.info(
            "Sluice: %d local experts -> %d GPU slots (%d protected), "
            "%d params/expert, %.1f MiB/expert.",
            local_n,
            num_slots,
            cache.protected_cap,
            len(per_expert),
            cache.bytes_per_expert / (1 << 20),
        )

    def _wrap_apply(self, module: nn.Module) -> None:
        """Wrap the layer's modular ``quant_method.apply`` so the routing-aware
        streaming hook fires before the fused-MoE kernel (and, when a step
        selects more experts than fit the cache, the kernel runs once per
        wave). This is what makes Sluice a plugin: no edit to vLLM's MoE
        runner source is required, since ``apply`` already receives ``layer``
        and ``topk_ids``."""
        quant_method = module.quant_method
        if quant_method is None or getattr(quant_method, "_sluice_wrapped", False):
            return
        if self._dp_mode:
            mk = self._find_mk(module)
            if mk is not None:
                # Under DP the pre-dispatch apply hook would stream for THIS
                # rank's routing while the kernel computes the gathered token
                # set — hook the modular kernel's post-dispatch seam instead.
                self._wrap_mk(module, mk)
                quant_method._sluice_wrapped = True
                return
        original_apply = quant_method.apply
        offloader = self

        def apply(*args, **kwargs):
            layer = kwargs.get("layer", args[0] if args else None)
            topk_ids = kwargs.get("topk_ids")
            if layer is None or topk_ids is None:
                return original_apply(*args, **kwargs)
            return offloader.run_moe(original_apply, args, kwargs, layer, topk_ids)

        quant_method.apply = apply
        quant_method._sluice_wrapped = True

    @staticmethod
    def _find_mk(module: nn.Module):
        """Locate the object exposing the modular-kernel dispatch seams
        (``_prepare`` + ``_fused_experts``), or None.

        Two generations exist in vLLM 0.23: internal-MK quant methods hold a
        ``FusedMoEKernel`` whose seams live on ``.impl``
        (FusedMoEKernelModularImpl); FusedMoEModularMethod holds a
        ``FusedMoEModularKernel`` with the seams directly on it. Duck-typing
        (not isinstance) — vLLM's lazy loaders can yield distinct class
        objects for the same source class. A monolithic impl exposes no
        ``_prepare`` and correctly returns None (no safe seam)."""
        qm = getattr(module, "quant_method", None)
        if qm is None:
            return None
        cands = []
        mk = getattr(qm, "moe_kernel", None)
        if mk is not None:
            cands.append(getattr(mk, "impl", None))
            cands.append(mk)
        cands.append(getattr(qm, "fused_experts", None))
        for c in cands:
            if (
                c is not None
                and callable(getattr(c, "_prepare", None))
                and callable(getattr(c, "_fused_experts", None))
            ):
                return c
        return None

    def _wrap_mk(self, module: nn.Module, mk) -> None:
        """Hook a modular kernel at its post-dispatch seam (DP-safe path).

        Dispatch (the DP/EP all-gather or all2all) happens inside
        ``mk._prepare``, which returns the POST-dispatch ``topk_ids`` — the
        ground truth for what the experts kernel is about to compute. The
        wrapped ``_prepare`` builds the streaming plan from those ids, fills
        wave 0 and writes its map; when the working set exceeds the cache the
        wrapped ``_fused_experts`` runs once per expert-wave, rewriting the
        map between calls and summing partials in fp32 (an unmapped expert
        contributes zero — the same algebra as the non-MK wave path; the MK
        experts kernel receives ``expert_map``, which is Sluice's live
        buffer, so in-place rewrites are what it reads).

        Policy is scan-only (no protected promotion): this path exists for
        DP, where per-rank row-layout decode signals are invalid anyway.
        Wave fills are not software-pipelined in this first version —
        correctness over overlap."""
        if getattr(mk, "_sluice_wrapped", False):
            return
        orig_prepare = mk._prepare
        orig_fused = mk._fused_experts
        offloader = self
        key = id(module)

        def _prepare(*a, **k):
            out = orig_prepare(*a, **k)
            # out = (a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_w)
            offloader._mk_plan_step(key, out[3])
            return out

        def _fused_experts(*fa, **fk):
            waves = offloader._mk_waves.get(key)
            if not waves or len(waves) == 1:
                return orig_fused(*fa, **fk)
            cache = offloader._caches[key]
            acc = None
            out = None
            for i, wave in enumerate(waves):
                if i:
                    offloader._mk_fill_wave(cache, wave)
                    offloader._write_map(
                        cache, [(g, s) for g, _l, s in wave], pinned=False
                    )
                out = orig_fused(*fa, **fk)
                acc = out.float() if acc is None else acc.add_(out.float())
            # Restore wave 0's map so a chunked forward (multiple
            # _fused_experts calls per step) replays the wave sequence from a
            # consistent starting map.
            offloader._write_map(
                cache, [(g, s) for g, _l, s in waves[0]], pinned=False
            )
            return acc.to(out.dtype)

        mk._prepare = _prepare
        mk._fused_experts = _fused_experts
        mk._sluice_wrapped = True
        self._mk_experts_name[key] = type(
            getattr(mk, "fused_experts", None)
        ).__name__

    def _mk_plan_step(self, key: int, post_ids: torch.Tensor) -> None:
        """Build this step's wave plan from POST-dispatch topk_ids: stream
        wave 0's misses, write its map, stash later waves for the wrapped
        ``_fused_experts``."""
        cache = self._caches.get(key)
        if cache is None or cache.static_full:
            return
        ids_cpu = post_ids.to("cpu")
        selected = torch.unique(ids_cpu).tolist()
        local_of = cache.local_of
        n_global = len(local_of)
        pairs = []
        for g in selected:
            if 0 <= g < n_global:
                local = local_of[g]
                if local >= 0:
                    pairs.append((g, local))
        copy_stream = (
            None if self._mk_sync_fills else self._get_copy_stream(cache.device)
        )
        if pairs and copy_stream is not None:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            copy_stream.wait_event(ev)
        hits: list[tuple[int, int, int]] = []
        missing: list[tuple[int, int]] = []
        for g, local in pairs:
            slot = cache.slot_of.get(local)
            if slot is None:
                missing.append((g, local))
            else:
                self._touch(cache, slot, False)
                hits.append((g, local, slot))
        claimed = {s for _g, _l, s in hits}
        wave0 = list(hits)
        n_fills = 0
        for g, local in missing:
            slot = self._acquire_slot(cache, local, claimed, allow_protected=False)
            if slot is None:
                break
            self._stream_in(cache, local, slot, copy_stream)
            claimed.add(slot)
            wave0.append((g, local, slot))
            n_fills += 1
        leftover = missing[n_fills:]
        waves = [wave0]
        if leftover:
            # MK waves are validated on both kernel classes (bit-identical:
            # V2-Lite triton DP=2 / DP=2xTP=2; clean: fp8-marlin DP=2 /
            # DP=2xTP=2 and V4-Pro DP=2xTP=2). The one measured
            # incompatibility is vLLM's custom fusion passes
            # (fuse_allreduce_rms / norm_quant / act_quant, flashinfer
            # allreduce) racing with wave-looped MoE under DP — a CUDA
            # illegal access; disable them under DP (_check_config warns
            # with the exact flags). SLUICE_MK_WAVES=0 remains the
            # kill-switch: with it set, refuse instead of waving.
            if os.environ.get("SLUICE_MK_WAVES", "1") == "0":
                raise RuntimeError(
                    f"Sluice: this step selected {len(pairs)} local experts "
                    f"> {cache.num_slots} slots and SLUICE_MK_WAVES=0 "
                    "forbids waves. Raise SLUICE_SLOTS or cap "
                    "--max-num-batched-tokens."
                )
            # Rotation window for the remaining waves. No pipelining, so any
            # slots may be reused once the previous wave's kernel is ordered
            # ahead of the fills (see _mk_fill_wave); prefer probation slots
            # to keep bookkeeping simple.
            window = list(cache.probation)
            if not window:
                window = list(range(cache.num_slots))[
                    -max(1, cache.num_slots // 4):
                ]
            w = len(window)
            for k in range(0, len(leftover), w):
                chunk = leftover[k:k + w]
                waves.append(
                    [(g, local, window[j]) for j, (g, local) in enumerate(chunk)]
                )
        self._mk_waves[key] = waves
        self._write_map(cache, [(g, s) for g, _l, s in wave0], pinned=True)
        if n_fills:
            self._compute_wait_copies(copy_stream)
        if self._diag:
            stats = self._stats["scan"]
            stats.steps += 1
            stats.hits += len(hits)
            stats.misses += len(missing)
            stats.bytes += len(missing) * cache.bytes_per_expert
            stats.waves += len(waves)
            # drives the working-set HWM log and the periodic summary printer
            # (real steps only: dummy/profiling runs have garbage routing and
            # would pollute the high-water marks)
            if self._split_signal()[2]:
                self._record_working_set(int(ids_cpu.shape[0]), len(pairs))

    def _mk_fill_wave(self, cache: _ExpertLayerCache, wave: list) -> None:
        """Fill a later wave's slots. The copies are ordered behind everything
        already enqueued on the compute stream (the previous wave's kernel may
        still be reading these slots), and the compute stream then waits for
        the fills before the next kernel launch."""
        copy_stream = (
            None if self._mk_sync_fills else self._get_copy_stream(cache.device)
        )
        if copy_stream is not None:
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            copy_stream.wait_event(ev)
        for _g, local, slot in wave:
            if cache.expert_in_slot[slot] != local:
                self._evict_to(cache, slot, local)
                self._stream_in(cache, local, slot, copy_stream)
            if slot in cache.protected:
                del cache.protected[slot]
                cache.probation[slot] = None
            elif slot not in cache.probation:
                cache.probation[slot] = None
        self._compute_wait_copies(copy_stream)

    # -- forward path (fired from the wrapped quant_method.apply) ------------

    def run_moe(self, original_apply, args, kwargs, module, topk_ids):
        cache = self._caches.get(id(module))
        if cache is None:
            return original_apply(*args, **kwargs)
        if cache.static_full:
            # Full residency: the identity map is already installed and never
            # changes, so skip the whole routing hook (no D2H sync, no unique,
            # no map rewrite) — the kernel runs exactly as a resident EP rank.
            return original_apply(*args, **kwargs)
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            # Unreachable when _check_config passed; a captured hook would
            # replay a frozen expert map, so refuse loudly rather than
            # generate plausible-but-wrong tokens.
            raise RuntimeError(
                "Sluice: streaming hook reached under CUDA graph capture; "
                "run with enforce_eager."
            )
        assert cache.map_host is not None and cache.expert_map_buf is not None

        # One host sync per layer: the router's decisions gate everything.
        # A single D2H copy, then unique on CPU (slicing decode vs prefill
        # rows costs no extra sync).
        nd, ndec, real_step = self._split_signal()
        ids_cpu = topk_ids.to("cpu")
        num_tokens = int(ids_cpu.shape[0])

        # Pure-prefill chunk (no decode tokens): the step needs ~all local
        # experts, so stream the whole shard in large contiguous blocks —
        # far higher effective PCIe bandwidth than per-expert copies, and no
        # decode-hot set to protect. Mixed prefill+decode steps fall through
        # to the SLRU path so a scan cannot evict the decode set.
        if (
            self.prefill_blocks
            and real_step
            and nd == 0
            and num_tokens >= 2 * cache.num_slots
            and cache.num_slots < cache.local_num_experts
        ):
            if self._diag:
                self._record_working_set(num_tokens, cache.local_num_experts)
            stats = self._stats["scan"]
            stats.steps += 1
            return self._run_prefill_blocks(cache, original_apply, args, kwargs, stats)

        selected = torch.unique(ids_cpu).tolist()
        # Only treat the [0:nd) region as decode-hot (protected-eligible) when
        # it is genuinely all single-token decodes: nd == ndec. When nd > ndec
        # the region contains short prefill "extends" vLLM folded into the
        # decode count; tagging those experts decode-class would let a short
        # prefill churn the protected set (the failure the design forbids), so
        # fall back to scan (decode_ids=None) — output stays correct via waves,
        # and the existing protected set survives (scan can't evict it).
        # Genuine decode region: ndec requests each verifying (1 + k) tokens
        # (k = speculative draft depth; k=0 without spec-decode → nd == ndec).
        # If nd doesn't match, short prefill "extends" are folded in and the
        # region can't be trusted for protection → fall back to scan.
        decode_span = ndec * (1 + self._spec_tokens) if ndec is not None else None
        pure_decode_region = (
            real_step and nd is not None and decode_span is not None
            and nd == decode_span and not self._dp_mode
        )
        if pure_decode_region and 0 < nd <= num_tokens:
            # rows [0:nd) are one-token decodes (vLLM reorders decode-first).
            decode_ids = (
                set(selected)
                if nd == num_tokens
                else set(torch.unique(ids_cpu[:nd]).tolist())
            )
        else:
            decode_ids = None

        # Non-MLA backends (FlashAttention-style) never reorder the batch and
        # publish no decode count — but a request with query_len == 1 IS a
        # decode (none of MLA's short-extend ambiguity), and its single row
        # index is its query_start_loc entry. Recovering the exact decode rows
        # lets mixed decode+prefill steps promote the decode-hot set instead
        # of degrading to the ws heuristic (which never promotes on mixed
        # steps, collapsing SLRU to plain LRU under continuous batching).
        non_mla_pure_decode = False
        if (
            decode_ids is None
            and real_step
            and nd is None
            and self.nonmla_classifier
            and not self._dp_mode
        ):
            rows = self._non_mla_decode_rows(num_tokens)
            if rows is not None:
                if len(rows) == num_tokens:
                    decode_ids = set(selected)
                    non_mla_pure_decode = True
                elif len(rows):
                    decode_ids = set(torch.unique(ids_cpu[rows]).tolist())
                else:
                    decode_ids = set()  # pure prefill: everything scan-class

        pairs = []  # (global_id, local_id, decode_class)
        local_of = cache.local_of
        n_global = len(local_of)
        heuristic_friendly = None
        for g in selected:
            if 0 <= g < n_global:
                local = local_of[g]
                if local >= 0:
                    pairs.append((g, local, decode_ids is not None and g in decode_ids))
        ws = len(pairs)
        # The ws<=protected_cap heuristic is a best-effort decode signal for
        # non-MLA backends (no decode count at all). Do NOT apply it when MLA
        # metadata is present but told us the region isn't pure decode
        # (nd > ndec, short prefills mixed in) — that would re-introduce the
        # protected-set churn we just guarded against; force scan instead.
        mla_forced_scan = real_step and nd is not None and not pure_decode_region
        if decode_ids is None and not mla_forced_scan and not self._dp_mode:
            # No per-token signal (dummy run, or non-MLA fallback): dummy runs
            # are pure scans; real steps fall back to the working-set size.
            heuristic_friendly = real_step and ws <= max(cache.protected_cap, 1)
            if heuristic_friendly:
                pairs = [(g, local, True) for g, local, _ in pairs]
        if self._diag and real_step:
            self._record_working_set(num_tokens, ws)

        if ws == 0:
            self._write_map(cache, [], pinned=True)
            return original_apply(*args, **kwargs)

        # Per-expert policy: decode-selected experts may promote into (and,
        # on miss, evict from) the protected segment; prefill-only/scan
        # experts are confined to probation so a chunked-prefill scan can
        # never flush the decode-hot set. Dummy runs (garbage routing) get
        # scan treatment and no stats.
        stats = None
        if real_step:
            if decode_ids is not None:
                pure_decode = (
                    (nd == num_tokens) if nd is not None else non_mla_pure_decode
                )
            else:
                pure_decode = heuristic_friendly
            stats = self._stats["small" if pure_decode else "scan"]
            stats.steps += 1
            if self.lfu:
                self._bump_freq(cache, pairs)

        hits: list[tuple[int, int]] = []  # (g, slot)
        missing: list[tuple[int, int, bool]] = []  # (g, local, decode_class)
        for g, local, dcls in pairs:
            slot = cache.slot_of.get(local)
            if slot is None:
                missing.append((g, local, dcls))
            else:
                self._touch(cache, slot, dcls)
                hits.append((g, slot))
        if stats is not None:
            stats.hits += len(hits)
            stats.misses += len(missing)
            stats.bytes += len(missing) * cache.bytes_per_expert

        # Wave 0: all hits, plus as many misses as we can stream in without
        # evicting a slot this step already relies on. Decode-class misses
        # get slots first so decode experts never spill into later waves.
        missing.sort(key=lambda t: not t[2])
        claimed = {slot for _, slot in hits}
        wave0 = list(hits)
        copy_stream = self._get_copy_stream(cache.device)
        if missing and copy_stream is not None:
            # Explicit compute->copy edge: fills may overwrite a slot the
            # previous kernel read. The per-layer host sync above already
            # drained the compute stream, but the edge must not depend on it.
            ev = torch.cuda.Event()
            ev.record(torch.cuda.current_stream())
            copy_stream.wait_event(ev)
        taken = len(missing)
        for i, (g, local, dcls) in enumerate(missing):
            slot = self._acquire_slot(cache, local, claimed, allow_protected=dcls)
            if slot is None:
                taken = i
                break
            self._stream_in(cache, local, slot, copy_stream)
            wave0.append((g, slot))
            claimed.add(slot)
        leftover = [(g, local) for g, local, _ in missing[taken:]]

        if not leftover:
            if stats is not None:
                stats.waves += 1
            self._write_map(cache, wave0, pinned=True)
            if taken:  # streamed something: kernel must wait for the copies
                self._compute_wait_copies(copy_stream)
            return original_apply(*args, **kwargs)

        if real_step and not self._warned_waves:
            self._warned_waves = True
            logger.warning(
                "Sluice: step selected %d local experts > usable slots; "
                "executing in waves (exact in math, not bit-identical to a "
                "single launch). Raise SLUICE_SLOTS to avoid waves.",
                ws,
            )
        return self._run_waves(
            cache, original_apply, args, kwargs, wave0, leftover, stats, copy_stream
        )

    @staticmethod
    def _split_signal() -> tuple[int | None, int | None, bool]:
        """Return ``(num_decode_tokens, num_decodes, is_real_step)`` from the v1
        forward context. ``(None, None, False)`` = dummy/profiling run;
        ``(None, None, True)`` = real step but no MLA decode split available
        (non-MLA backend); ``(nd, ndec, True)`` = rows [0:nd) are the decode
        region, spanning ``ndec`` requests.

        NOTE: vLLM's ``num_decode_tokens`` counts short prefill "extends"
        (query_len <= the MLA reorder threshold, 512/128) as decode. So the
        [0:nd) region is genuinely all single-token decodes ONLY when
        ``nd == ndec``; when ``nd > ndec`` it contains mis-classified short
        prefills, and the caller must not treat it as protected-eligible."""
        try:
            from vllm.forward_context import get_forward_context

            ctx = get_forward_context()
        except Exception:
            return None, None, False
        md = getattr(ctx, "attn_metadata", None)
        if md is None:
            return None, None, False
        if isinstance(md, dict):
            md = next(iter(md.values()), None)
            if md is None:
                return None, None, False
        nd = getattr(md, "num_decode_tokens", None)
        if nd is None:
            return None, None, True
        ndec = getattr(md, "num_decodes", None)
        try:
            return int(nd), (int(ndec) if ndec is not None else None), True
        except (TypeError, ValueError):
            return None, None, True

    @staticmethod
    def _non_mla_decode_rows(num_tokens: int) -> "torch.Tensor | None":
        """Exact decode-row indices for non-MLA v1 backends (FlashAttention/
        Triton-style metadata).

        These backends don't reorder the batch or publish a decode count, but
        ``query_start_loc`` gives per-request query lengths, and a request
        with query_len == 1 is a decode whose one token row is its start
        index — unlike MLA's ``num_decode_tokens`` there is no short-extend
        ambiguity to guard. The D2H copy here is a few hundred bytes and the
        step already synced for ``topk_ids``.

        Returns None when the metadata doesn't line up (caller falls back to
        the ws heuristic); an empty tensor means pure prefill (all scan)."""
        try:
            from vllm.forward_context import get_forward_context

            md = get_forward_context().attn_metadata
        except Exception:
            return None
        if isinstance(md, dict):
            md = next(iter(md.values()), None)
        if md is None:
            return None
        mql = getattr(md, "max_query_len", None)
        qsl = getattr(md, "query_start_loc", None)
        if mql is None or qsl is None or not isinstance(qsl, torch.Tensor):
            return None
        nat = getattr(md, "num_actual_tokens", None)
        if nat is not None and int(nat) != num_tokens:
            return None  # padding or a batch shape we don't understand
        if int(mql) == 1:
            return torch.arange(num_tokens)  # pure decode
        qsl_cpu = qsl.to("cpu")
        if (
            qsl_cpu.ndim != 1
            or qsl_cpu.numel() < 2
            or int(qsl_cpu[-1]) != num_tokens
        ):
            return None
        starts = qsl_cpu[:-1]
        qlens = qsl_cpu[1:] - starts
        return starts[qlens == 1].long()

    def _run_waves(
        self, cache, original_apply, args, kwargs, wave0, leftover, stats, copy_stream
    ):
        """Execute a step whose working set exceeds the cache: wave 0 (hits +
        streamed fills), then the remaining misses through probation slots in
        ping-pong groups, software-pipelined so wave k+1's H2D copies run on
        the copy stream while wave k's kernel computes. Output = sum of wave
        outputs, which is exact because unmapped experts contribute zero."""
        # Rotation groups: probation slots split in two (ping-pong). Protected
        # slots are never touched, so the decode-hot set survives the scan.
        rot = list(cache.probation)
        if not rot:  # degenerate: no probation slots — rotate a small tail
            rot = list(range(cache.num_slots))[-max(1, cache.num_slots // 4):]
        half = (len(rot) + 1) // 2
        group_a, group_b = rot[:half], rot[half:]
        pipelined = bool(group_b)
        if not pipelined:
            group_b = group_a
        groups = (group_a, group_b)

        # Plan waves 1..n (bookkeeping only; copies are enqueued just-in-time
        # inside the execution loop below, after the kernel events they must
        # wait on exist).
        waves: list[list[tuple[int, int]]] = [wave0]
        chunks: list[list[tuple[int, int, int]]] = [[]]  # (g, local, slot)
        i = 0
        gi = 0
        while i < len(leftover):
            group = groups[gi % 2]
            chunk = []
            for (g, local), slot in zip(leftover[i : i + len(group)], group):
                self._evict_to(cache, slot, local)
                chunk.append((g, local, slot))
            waves.append([(g, slot) for g, _, slot in chunk])
            chunks.append(chunk)
            i += len(chunk)
            gi += 1
        if stats is not None:
            stats.waves += len(waves)

        compute = torch.cuda.current_stream()
        kernel_ev: list[torch.cuda.Event] = []
        copy_ready: dict[int, torch.cuda.Event] = {}

        def enqueue_copies(k: int) -> None:
            """Enqueue wave k's H2D copies (k >= 1) on the copy stream, after
            the last kernel that read this rotation group's slots: wave k-2
            when ping-ponging (so copies overlap wave k-1's kernel), else the
            immediately preceding wave. Wave 0's kernel maps every slot, so it
            is the floor dependency."""
            dep = kernel_ev[k - 1 if not pipelined else max(k - 2, 0)]
            if copy_stream is not None:
                with torch.cuda.stream(copy_stream):
                    copy_stream.wait_event(dep)
                    for _g, local, slot in chunks[k]:
                        self._stream_in(cache, local, slot, None, on_stream=True)
                    ev = torch.cuda.Event()
                    ev.record(copy_stream)
                    copy_ready[k] = ev
            else:
                # No copy stream: copies go on the compute stream, which is
                # already ordered after every prior kernel.
                for _g, local, slot in chunks[k]:
                    self._stream_in(cache, local, slot, None, on_stream=True)

        total = None
        for k, wave in enumerate(waves):
            if k == 0:
                self._write_map(cache, wave, pinned=True)
                self._compute_wait_copies(copy_stream)
            else:
                self._write_map(cache, wave, pinned=False)
                if copy_stream is not None:
                    compute.wait_event(copy_ready[k])
            out = original_apply(*args, **kwargs)
            ev = torch.cuda.Event()
            ev.record(compute)
            kernel_ev.append(ev)
            if k + 1 < len(waves):
                enqueue_copies(k + 1)
            # Accumulate in fp32: wave partials are exact in math but change
            # float summation order; fp32 keeps the drift vs a single launch
            # at the ulp level.
            total = out.to(torch.float32) if total is None else total.add_(out)
        return total.to(out.dtype)

    def _run_prefill_blocks(self, cache, original_apply, args, kwargs, stats):
        """Pure-prefill fast path: stream the whole local expert shard in
        contiguous blocks of ``num_slots`` (one large H2D per param per block,
        vs one small copy per expert), running the kernel once per block and
        summing. Correct because an expert no token routes to contributes zero
        — exposing a whole contiguous block per launch matches how a resident
        EP rank exposes all its local experts. All work is on the compute
        stream, so each block's fill is ordered after the previous block's
        kernel (no cross-stream write-after-read hazard on the reused slots)."""
        local_n = cache.local_num_experts
        slots = cache.num_slots
        gol = cache.global_of_local
        n_blocks = (local_n + slots - 1) // slots
        if stats is not None:
            stats.waves += n_blocks
            stats.bytes += local_n * cache.bytes_per_expert
        total = None
        out = None
        last_lo = last_k = 0
        for w in range(n_blocks):
            lo = w * slots
            k = min(slots, local_n - lo)
            for name, gpu in cache.gpu_cache.items():
                gpu[:k].copy_(cache.cpu_store[name][lo : lo + k], non_blocking=True)
            wave = [(gol[lo + j], j) for j in range(k) if gol[lo + j] >= 0]
            self._write_map(cache, wave, pinned=False)
            out = original_apply(*args, **kwargs)
            total = out.to(torch.float32) if total is None else total.add_(out)
            last_lo, last_k = lo, k
        # Reset SLRU bookkeeping to the block left resident, so a following
        # decode step reads correct slots and rebuilds its protected set.
        cache.slot_of.clear()
        cache.protected.clear()
        cache.probation.clear()
        cache.expert_in_slot = [None] * slots
        cache.free_slots = list(range(slots - 1, last_k - 1, -1))
        for j in range(last_k):
            cache.slot_of[last_lo + j] = j
            cache.expert_in_slot[j] = last_lo + j
            cache.probation[j] = None
        return total.to(out.dtype)

    # -- cache/SLRU internals ------------------------------------------------

    def _get_copy_stream(self, device) -> "torch.cuda.Stream | None":
        if device.type != "cuda":
            return None
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=device)
        return self._copy_stream

    def _compute_wait_copies(self, copy_stream) -> None:
        if copy_stream is None:
            return
        ev = torch.cuda.Event()
        ev.record(copy_stream)
        torch.cuda.current_stream().wait_event(ev)

    def _bump_freq(self, cache: _ExpertLayerCache, pairs: list) -> None:
        """Accumulate per-expert popularity and periodically age it (halve),
        so LFU tracks the current hot set rather than startup transients."""
        freq = cache.freq
        for _g, local, _d in pairs:
            freq[local] = freq.get(local, 0) + 1
        cache.lfu_steps += 1
        if cache.lfu_steps >= LFU_AGE_STEPS:
            cache.lfu_steps = 0
            for k in list(freq.keys()):
                v = freq[k] >> 1
                if v:
                    freq[k] = v
                else:
                    del freq[k]

    def _touch(self, cache: _ExpertLayerCache, slot: int, friendly: bool) -> None:
        if slot in cache.protected:
            cache.protected.move_to_end(slot)
        elif slot in cache.probation:
            if friendly and cache.protected_cap > 0:
                del cache.probation[slot]
                cache.protected[slot] = None
                while len(cache.protected) > cache.protected_cap:
                    demoted = self._coldest_protected(cache)
                    del cache.protected[demoted]
                    cache.probation[demoted] = None
            else:
                cache.probation.move_to_end(slot)

    def _coldest_protected(self, cache: _ExpertLayerCache) -> int:
        """Victim to demote from protected: least-frequently-used under LFU
        (ties broken by LRU order), else plain LRU."""
        if self.lfu and cache.freq:
            freq = cache.freq
            in_slot = cache.expert_in_slot
            # protected iterates LRU-first, so min() keeps the LRU tiebreak.
            return min(
                cache.protected,
                key=lambda s: freq.get(in_slot[s], 0),
            )
        return next(iter(cache.protected))

    def _acquire_slot(
        self,
        cache: _ExpertLayerCache,
        local: int,
        claimed: set,
        allow_protected: bool,
    ) -> int | None:
        """Find a slot for a missing expert: free list, then probation LRU,
        then (cache-friendly steps only) protected LRU. Never a slot this
        step already claimed."""
        slot = None
        if cache.free_slots:
            slot = cache.free_slots.pop()
        else:
            for cand in cache.probation:
                if cand not in claimed:
                    slot = cand
                    del cache.probation[slot]
                    break
            if slot is None and allow_protected:
                # LRU by default; under LFU walk protected coldest-first so a
                # frequently-used decode expert is never the victim.
                order = cache.protected
                if self.lfu and cache.freq:
                    freq = cache.freq
                    in_slot = cache.expert_in_slot
                    order = sorted(
                        cache.protected, key=lambda s: freq.get(in_slot[s], 0)
                    )
                for cand in order:
                    if cand not in claimed:
                        slot = cand
                        del cache.protected[slot]
                        break
        if slot is None:
            return None
        self._evict_to(cache, slot, local)
        cache.probation[slot] = None  # new entries start probationary
        return slot

    def _evict_to(self, cache: _ExpertLayerCache, slot: int, local: int) -> None:
        old = cache.expert_in_slot[slot]
        if old is not None:
            cache.slot_of.pop(old, None)
        cache.expert_in_slot[slot] = local
        cache.slot_of[local] = slot

    def _write_map(self, cache: _ExpertLayerCache, wave: list, pinned: bool) -> None:
        """Point the layer's expert map at this wave's slots (-1 elsewhere).
        The steady-state single-wave path reuses the pinned host buffer with an
        async copy; extra waves build a fresh host tensor and copy it
        synchronously so an in-flight copy of a previous wave's map is never
        mutated under it (the sync also keeps CPU dispatch from outrunning the
        kernel it must order behind — copies for the wave after next were
        already enqueued, so overlap is preserved)."""
        assert cache.map_host is not None and cache.expert_map_buf is not None
        if pinned:
            if cache.map_ev is not None:
                cache.map_ev.synchronize()  # prior async copy of this buffer
            host = cache.map_host
            host.fill_(-1)
        else:
            host = torch.full(
                cache.map_host.shape, -1, dtype=cache.map_host.dtype, device="cpu"
            )
        for g, slot in wave:
            host[g] = slot
        cache.expert_map_buf.copy_(host, non_blocking=pinned)
        if pinned and cache.expert_map_buf.is_cuda:
            if cache.map_ev is None:
                cache.map_ev = torch.cuda.Event()
            cache.map_ev.record(torch.cuda.current_stream())

    def _stream_in(
        self,
        cache: _ExpertLayerCache,
        local: int,
        slot: int,
        copy_stream,
        on_stream: bool = False,
    ) -> None:
        if on_stream or copy_stream is None:
            for name, gpu in cache.gpu_cache.items():
                gpu[slot].copy_(cache.cpu_store[name][local], non_blocking=True)
            return
        with torch.cuda.stream(copy_stream):
            for name, gpu in cache.gpu_cache.items():
                gpu[slot].copy_(cache.cpu_store[name][local], non_blocking=True)

    # -- diagnostics ----------------------------------------------------------

    def _record_working_set(self, num_tokens: int, distinct: int) -> None:
        """Log new high-water marks of distinct local experts selected in one
        step, per step-size bucket, plus periodic hit-rate summaries."""
        if distinct > self._ws_hwm.get(num_tokens, 0):
            self._ws_hwm[num_tokens] = distinct
            # WARNING level: vLLM suppresses INFO from non-vllm loggers, but
            # surfaces WARNING (same path as the overflow warning).
            logger.warning(
                "Sluice DIAG: step tokens=%d -> %d distinct local experts "
                "(per-layer high-water; fits in one wave iff <= slots).",
                num_tokens,
                distinct,
            )
        self._hook_calls += 1
        if self._stats_every > 0 and self._hook_calls % self._stats_every == 0:
            for kind, s in self._stats.items():
                if not s.steps:
                    continue
                denom = max(s.hits + s.misses, 1)
                logger.warning(
                    "Sluice DIAG[%s]: steps=%d hit=%.1f%% streamed=%.2f GiB "
                    "waves/step=%.2f",
                    kind,
                    s.steps,
                    100.0 * s.hits / denom,
                    s.bytes / (1 << 30),
                    s.waves / s.steps,
                )

    def get_stats(self) -> dict:
        return {
            kind: {
                "steps": s.steps,
                "hits": s.hits,
                "misses": s.misses,
                "bytes": s.bytes,
                "waves": s.waves,
            }
            for kind, s in self._stats.items()
        }
