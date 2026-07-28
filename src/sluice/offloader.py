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

Graph modes
-----------
Default is eager (enforced at config time — a captured graph would freeze one
step's expert map and replay stale routing). ``SLUICE_GRAPH`` permits full
capture only at full residency (every layer static_full: the map is a constant
identity). ``SLUICE_PIECEWISE`` keeps streaming but runs the hook eagerly in a
dynamo graph break while attention/norms are captured as piecewise CUDA
graphs. ``SLUICE_ROUTER_SPLIT`` (requires piecewise) splits *inside* the MoE
layer: gate + select_experts and the fused-experts GEMM are captured, with one
thin eager gap (``vllm::sluice_stream_gap``) between them that streams the
step's missing experts and refreshes the expert map — graphs bake POINTERS,
the gap rewrites CONTENTS. See ``attach_router_split``.
"""

import atexit
import itertools
import json
import os
import time
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
# Opt-in profiling: dump cumulative per-expert selection counts (EPLB-style
# load statistics) as JSONL snapshots — the measurement half of a static
# hot-expert pin. SLUICE_EXPERT_COUNTS=<dir> enables it (one file per rank);
# SLUICE_EXPERT_COUNTS_EVERY sets the snapshot period in hook calls.
COUNTS_DIR_ENV = "SLUICE_EXPERT_COUNTS"
COUNTS_EVERY_ENV = "SLUICE_EXPERT_COUNTS_EVERY"
# Opt-in static hot-expert pin: JSON file {"layers": {"<layer_ord>": [global
# expert ids, ...]}} where layer_ord is the MoE-layer install order (the same
# ordinal the SLUICE_EXPERT_COUNTS dumps use). Per layer, listed experts owned
# by this rank are streamed into slots once at init and never evicted; the pin
# budget is deducted from the protected segment so probation (the prefill
# scan buffer) keeps its exact unpinned size. Derive the file from counts
# dumps with examples/derive_pin_set.py.
PIN_FILE_ENV = "SLUICE_PIN_FILE"
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
    # Statically pinned slots (PIN_FILE_ENV): filled at init, never evicted.
    # They sit in NEITHER SLRU segment nor the free list, so _touch no-ops on
    # them (hits cost nothing) and _acquire_slot can never pick them.
    pinned_slots: set = field(default_factory=set)
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
    # Hook-lite: True when expert_map_buf currently maps EVERY resident expert
    # to its slot (a "standing" map), so an all-hit step needs no rewrite. Any
    # sparse/wave map write clears it (see _write_map); _write_standing_map
    # sets it.
    map_is_standing: bool = False
    # Lazy-sync: device bool mask over global ids, True where this rank owns
    # the expert — for exact miss counting on the sync-free path.
    owned_mask_dev: "torch.Tensor | None" = None


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
        # Hook-lite (opt-in): keep a STANDING expert map covering every
        # resident expert (pinned + cached), so an all-hit single-wave step
        # needs no per-layer map rewrite and no map_ev stall — the kernel runs
        # against the already-correct map. Semantically identical to the
        # legacy path (same evictions/promotions/output; the standing map is a
        # superset of the sparse map, and non-selected mapped experts receive
        # no tokens), it only elides the redundant H2D on hit steps. Targets
        # the decode-solely / high-residency regime where the per-layer sync
        # tax dominates. Does NOT remove the topk_ids D2H sync (that is
        # SLUICE_LAZY_SYNC); measures how much of the hook floor is the map
        # rewrite.
        self.hook_lite = os.environ.get("SLUICE_HOOK_LITE", "0") == "1"
        # Lazy-sync (opt-in, measurement-grade): the NEXT step past hook-lite.
        # When the standing map covers every selected expert (guaranteed at
        # full residency), an all-hit decode step needs nothing from the CPU —
        # the map is already on the GPU and topk_ids is already on the GPU, so
        # the kernel runs with NO per-layer topk_ids D2H sync, no unique/
        # classify/pairs, no map write. This is the path that kills the ~60%
        # of the hook tax that hook-lite v1 left (the sync + decision Python).
        # It is OPTIMISTIC: it engages when every local expert is resident (so
        # a miss is impossible), and a device miss counter verifies that held
        # (0 == the run was exactly correct). Partial-residency production use
        # needs a step-level fallback (not built here); this proves the ceiling.
        self.lazy_sync = os.environ.get("SLUICE_LAZY_SYNC", "0") == "1"
        if self.lazy_sync:
            self.hook_lite = True  # lazy-sync needs the standing-map machinery
        # Debug: force the streaming hook even at full residency (num_slots ==
        # local experts), which normally takes the static_full bypass. Lets the
        # sync-cost A/B (legacy vs hook-lite vs lazy-sync) run at a controlled
        # 0-miss operating point instead of silently short-circuiting.
        self.no_static_full = os.environ.get("SLUICE_NO_STATIC_FULL", "0") == "1"
        # Verify the 0-miss invariant on the lazy path with a per-layer device
        # miss counter. Correctness proof / telemetry only — it launches a few
        # extra small kernels per layer, so it costs a few % and is OFF by
        # default (the residency gate already guarantees correctness; the check
        # only proves it). Turn on for validation runs, off for the ceiling.
        self.lazy_miss_check = os.environ.get("SLUICE_LAZY_MISS_CHECK", "0") == "1"
        # Graph mode (opt-in, M1): permit non-eager / CUDA-graph capture. SAFE
        # ONLY when every layer is static_full — then the expert map is the
        # constant identity (written once at init, never mutated), so a captured
        # graph replays exactly correct routing. _check_config relaxes the eager
        # guards under this flag; post_init then ASSERTS all-static_full and
        # refuses otherwise (fail closed — a graph over the streaming hook would
        # bake a stale map and be silently wrong).
        self.graph_mode = os.environ.get("SLUICE_GRAPH", "0") == "1"
        # Piecewise: keep offloading (slots < experts, hook runs) but force
        # the hook to a dynamo graph break so it executes EAGER in the gap
        # while attention/norms are captured as piecewise CUDA graphs —
        # recovers the non-MoE launch overhead without capturing the
        # streaming sync. Requires cudagraph_mode=PIECEWISE (FULL is refused).
        # Foundation for ROUTER-SPLIT below.
        self.piecewise = os.environ.get("SLUICE_PIECEWISE", "0") == "1"
        # GEMM-graph (kept negative result — PARKED after five distinct
        # capture hazards; superseded by ROUTER-SPLIT): capture a PRIVATE CUDA
        # graph of the fused-experts call (original_apply) per (layer,
        # token-count) bucket, and replay it on single-wave decode steps
        # instead of launching its kernels eagerly. The gap then costs sync +
        # stream + map-write + one replay. Static input buffers (copy-in per
        # step); weights/map are read via their live pointers (contents may
        # change — a graph bakes pointers, not values). Multi-wave/prefill/
        # oversized steps fall back to the eager call. Any error disables it
        # for the process (fail open to eager, never wrong).
        # v2 target: the MODULAR KERNEL's forward (mk.forward) — the pure
        # routed GEMM, BELOW apply(): below the shared experts and their
        # aux-stream overlap that made apply()-level capture incorrect
        # (four-round forensics, see hook/piecewise_results.md). mk.forward is
        # single-stream tensor-in/tensor-out; weights are persistent instance
        # attrs; the expert map is read via its live pointer. The apply()-level
        # hazards (1)-(3) fixes carry over: side-effect copy-back, identity
        # staging with per-replay verification, real-step gating.
        self.gemm_graph = os.environ.get("SLUICE_GEMM_GRAPH", "0") == "1"
        self._gg_step = False  # set by run_moe around the eligible apply call
        # ROUTER-SPLIT (the validated graph-parity path): patch each
        # runner's _forward_entry — TRACED python — so decode-sized steps run
        #   select_experts [captured] -> sluice_stream_gap [the ONLY eager
        #   gap: sync+stream+map write] -> torch.ops.vllm.fused_experts
        #   [captured, reads slot-weights + map BY POINTER] -> shared MLP
        #   [captured] -> (shared, routed)
        # and the stock traced tail does all combine/scale/reduce (fidelity
        # inherited, not reimplemented). Prefill (> slots//topk tokens)
        # branches to the stock opaque path (classic hook, waves). vLLM's own
        # capture handles the GEMM + shared — the aux-stream hazard that broke
        # private capture never arises. Fidelity gate: under EAGER (compile
        # off) the traced python runs plain, same kernels => bit-identical to
        # stock, verified.
        self.router_split = os.environ.get("SLUICE_ROUTER_SPLIT", "0") == "1"
        self._rs_caches: list = []  # layer_idx -> cache (gap-op registry)
        self._rs_gaps = 0  # DIAG: gap invocations
        self._rs_misses = 0  # DIAG: experts streamed in from the gap
        self._rs_layers = 0  # DIAG: layers patched by attach_router_split
        # Single-wave branch boundary (tokens): set from the batch envelope at
        # config time under piecewise; attach_router_split derives a per-layer
        # fallback for eager runs.
        self._rs_thr = None
        # Perf diagnostic ONLY: gap returns without sync/stream/map-write, so
        # the arm measures the structural cost of the graph splits alone.
        # Outputs are INVALID (stale expert map).
        self._rs_noop = os.environ.get("SLUICE_RS_NOOP", "0") == "1"
        if self._rs_noop:
            logger.warning(
                "Sluice: SLUICE_RS_NOOP=1 — the router-split gap is a NO-OP; "
                "OUTPUTS ARE INVALID (stale expert map). Perf diagnostic "
                "only: never use this arm for quality or token comparison."
            )
        # All-hit fast path in the gap (exact; skips LRU bookkeeping on
        # no-miss steps, keeps the D2H sync).
        self._rs_fast_hit = os.environ.get("SLUICE_RS_FAST_HIT", "0") == "1"
        self._rs_fast_hits = 0
        # Staged-ids (kept negative result — measured 6-8% SLOWER than the
        # classic sync gap): a capturable async D2H of topk_ids into a pinned
        # per-layer buffer INSIDE the captured piece (right after select),
        # doorbell-stamped; the gap spin-waits the doorbell instead of
        # syncing the stream, overlapping PCIe latency with the piece's
        # remaining GPU work. Bounded-spin fallback to the classic sync
        # path self-heals capture passes (recorded, not executed).
        self._rs_staged = os.environ.get("SLUICE_RS_STAGED", "0") == "1"
        self._rs_stage: list = []  # layer_idx -> stage state
        self._rs_stage_falls = 0  # DIAG: bounded-spin fallbacks
        if self._rs_staged:
            logger.warning(
                "Sluice: SLUICE_RS_STAGED=1 — doorbell-staged topk_ids is a "
                "kept NEGATIVE result (measured 6-8% slower than the classic "
                "sync gap). Enable for re-measurement only."
            )
        if self.router_split:
            self._register_stream_gap_op()
        # LAZY-STEP (kept negative result — measured 12-17% SLOWER than the
        # classic hook): run the whole forward OPTIMISTICALLY sync-free —
        # every decode-sized layer whose standing map is valid skips the
        # topk_ids D2H sync/planning and just launches the kernel, while a
        # device counter accumulates misses (owned but unmapped selections).
        # ONE boundary check per forward (in the model.forward wrapper
        # installed by attach_model): zero misses → commit (the per-layer
        # syncs collapse to one); misses → re-run the forward with the
        # classic hooked path, which streams the misses and refreshes the
        # standing maps. The same-step rerun is exact: attention rewrites the
        # same KV slots for the same positions with corrected values, and
        # sampling happens once, after. Bit-identical to classic by
        # construction — but the rerun tax outweighs the saved syncs, so OFF.
        self.lazy_step = os.environ.get("SLUICE_LAZY_STEP", "0") == "1"
        if self.lazy_step:
            self.hook_lite = True  # standing maps are the foundation
            logger.warning(
                "Sluice: SLUICE_LAZY_STEP=1 — the optimistic sync-free "
                "forward is a kept NEGATIVE result (measured 12-17% slower "
                "than the classic hook: miss-reruns outweigh the saved "
                "syncs). Enable for re-measurement only."
            )
        self._lazy_now = False  # this forward is running the sync-free mode
        self._lazy_ran = False  # >=1 lazy layer executed this forward
        self._lazy_disabled = False
        self._lstep_stats = [0, 0, 0]  # clean, miss-rerun, classic forwards
        if self.gemm_graph:
            logger.warning(
                "Sluice: SLUICE_GEMM_GRAPH=1 — private mk-level capture is a "
                "kept NEGATIVE result, PARKED after five distinct capture "
                "hazards (see hook/piecewise_results.md); superseded by "
                "SLUICE_ROUTER_SPLIT. Enable for re-measurement only, gated "
                "by the bit-compare smoke."
            )
        self._gg: dict = {}  # (layer_key, ntok) -> capture entry
        # Self-healing classification: args caught changing identity after
        # being classified persistent (e.g. a reused-then-swapped output
        # buffer) get forced per-step here and the key's buckets recapture.
        self._gg_force_perstep: dict = {}  # layer_key -> {("a",i) | ("k",name)}
        self._gg_disabled = False
        self._gg_replays = 0
        self._gg_captures = 0
        self._lite_skips = 0  # all-hit steps that skipped the map rewrite
        self._lite_writes = 0  # standing-map (re)writes after residency change
        self._lite_marked = False
        self._lazy_steps = 0  # per-layer calls served by the sync-free path
        self._lazy_miss_dev = None  # device scalar: total misses on lazy path
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
        # Opt-in per-expert selection-count profiling (COUNTS_DIR_ENV): per
        # layer, two count vectors over GLOBAL expert ids (decode-class and
        # scan-class selections). This rank's shard only, so EP rank files
        # merge disjointly (each global id is counted by exactly one rank).
        self._counts_dir = os.environ.get(COUNTS_DIR_ENV)
        self._counts: dict[int, tuple[list[int], list[int]]] = {}
        self._counts_meta: dict[int, tuple[int, str]] = {}
        self._counts_calls = 0
        self._counts_snap = 0
        self._counts_file = None
        if self._counts_dir:
            try:
                self._counts_every = int(
                    os.environ.get(COUNTS_EVERY_ENV, "5000")
                )
            except ValueError:
                self._counts_every = 5000
            atexit.register(self._dump_expert_counts, True)
        # Static pin (opt-in): parsed once here, applied per layer in
        # _install_cache. A malformed or unreadable file fails loudly — a
        # silently empty pin would invalidate any experiment built on it.
        self._pin_layers: dict[int, list[int]] | None = None
        self._pin_stats = [0, 0]  # [experts pinned, layers with pins], this rank
        pin_path = os.environ.get(PIN_FILE_ENV)
        if pin_path:
            with open(pin_path) as f:
                raw = json.load(f)
            layers = raw.get("layers", raw)
            self._pin_layers = {
                int(k): [int(g) for g in v] for k, v in layers.items()
            }
            logger.info(
                "Sluice: static pin file %s (%d layers listed).",
                pin_path,
                len(self._pin_layers),
            )
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
        debug_moe = os.environ.get("SLUICE_DEBUG_MOE")
        for module in modules_generator:
            for sub in module.modules():
                if debug_moe:
                    self._debug_dump_moe(sub)
                if self._is_moe_layer(sub):
                    self._prepare_layer(sub)
            modules.append(module)
        return modules

    @staticmethod
    def _debug_dump_moe(module: nn.Module) -> None:
        """SLUICE_DEBUG_MOE: log the layout of any module that smells MoE (has
        a quant_method or an expert-count-ish attr) — attr names, values, and
        the recurse=False params with shapes — so a new model's expert layout
        can be diagnosed without guessing which attribute names it uses."""
        expert_attrs = [
            a for a in (
                "local_num_experts", "global_num_experts", "num_experts",
                "num_local_experts", "n_routed_experts", "num_experts_per_tok",
                "ep_size", "expert_map",
            ) if hasattr(module, a)
        ]
        if not expert_attrs and not hasattr(module, "quant_method"):
            return
        vals = {a: getattr(module, a, None) for a in expert_attrs
                if a not in ("expert_map",)}
        params = [
            (name, tuple(p.shape), str(p.dtype).replace("torch.", ""))
            for name, p in module.named_parameters(recurse=False)
        ]
        logger.warning(
            "SLUICE_DEBUG_MOE %s: attrs=%s qm=%s is_moe=%s params=%s",
            type(module).__name__, vals,
            type(getattr(module, "quant_method", None)).__name__,
            ExpertStreamOffloader._is_moe_layer(module), params,
        )

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
        # Fail closed: SLUICE_GRAPH permitted non-eager in _check_config on the
        # promise that every layer is static_full (constant identity map, safe
        # to capture). Enforce that promise now that the caches exist — a graph
        # over a streaming layer would replay stale routing (silently wrong).
        if self.graph_mode:
            streaming = [
                k for k, c in self._caches.items() if not c.static_full
            ]
            if streaming:
                raise RuntimeError(
                    "Sluice: SLUICE_GRAPH=1 requires every layer to be "
                    f"static_full (slots >= local experts), but {len(streaming)}"
                    " layer(s) are streaming. A captured graph would freeze a "
                    "changing expert map. Raise SLUICE_SLOTS to full residency "
                    "or unset SLUICE_GRAPH."
                )
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
            if self._pin_stats[0]:
                logger.info(
                    "Sluice: static pin active — %d experts pinned across %d "
                    "layers on this rank (budget taken from protected).",
                    self._pin_stats[0],
                    self._pin_stats[1],
                )
                # Filesystem engagement marker: sluice INFO logs are invisible
                # in some engine subprocess configs, and experiments must be
                # able to PROVE the pin engaged rather than trust a log line.
                pin_path = os.environ.get(PIN_FILE_ENV)
                if pin_path:
                    try:
                        try:
                            import torch.distributed as dist

                            rank = (
                                dist.get_rank() if dist.is_initialized() else 0
                            )
                        except Exception:
                            rank = 0
                        with open(
                            f"{pin_path}.applied.rank{rank}.json", "w"
                        ) as f:
                            json.dump(
                                {
                                    "pinned": self._pin_stats[0],
                                    "layers": self._pin_stats[1],
                                },
                                f,
                            )
                    except Exception:
                        logger.exception("Sluice: pin marker write failed")

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
        # SLUICE_GRAPH relaxes both eager guards: a captured graph is safe when
        # the map is constant, which holds iff every layer is static_full.
        # post_init asserts that after the caches exist (fail closed here would
        # be premature — caches are not installed yet).
        # SLUICE_PIECEWISE (experimental): permit non-eager PIECEWISE cudagraph.
        # The per-layer hook is forced to a dynamo graph break (torch._dynamo.
        # disable in _wrap_apply), so it runs EAGER in the gap while attention/
        # norms are captured — the streaming sync never enters a captured
        # region. FULL cudagraph is still refused (would capture the hook).
        relax_eager = self.graph_mode or self.piecewise
        mc = getattr(cfg, "model_config", None)
        if (
            mc is not None
            and not getattr(mc, "enforce_eager", False)
            and not relax_eager
        ):
            raise RuntimeError(
                "Sluice: requires eager execution — run with --enforce-eager. "
                "Without it, CUDA-graph capture freezes a stale expert map "
                "(silently wrong output) and stock torch.compile skips the "
                "post_init that installs Sluice's expert cache. "
                "(Set SLUICE_GRAPH=1 only with slots >= experts / static_full.)"
            )
        cc = getattr(cfg, "compilation_config", None)
        cg = getattr(cc, "cudagraph_mode", None) if cc is not None else None
        cg_name = getattr(cg, "name", str(cg)) if cg is not None else "NONE"
        if self.piecewise and "FULL" in cg_name:
            raise RuntimeError(
                "Sluice: SLUICE_PIECEWISE requires PIECEWISE cudagraph (the hook "
                f"runs eager in a graph break); got {cg_name}. FULL would try to "
                "capture the hook's D2H sync."
            )
        if (
            cg is not None
            and cg_name != "NONE"
            and not relax_eager
        ):
            raise RuntimeError(
                "Sluice: CUDA graphs would capture a frozen expert map and "
                "replay it with stale routing (silently wrong outputs). Run "
                "with enforce_eager (--enforce-eager)."
            )
        if self.piecewise and cc is not None:
            # Make the MoE an eager gap: vLLM dispatches the whole MoE layer
            # through the opaque custom ops vllm::moe_forward[_shared] (the
            # streaming hook runs INSIDE them). Adding them to splitting_ops
            # makes the fx splitter cut the compiled graph there, so the
            # captured pieces hold only attention/norms and the hook (with its
            # D2H sync and map writes) always executes eagerly in the gap.
            # This runs at offloader construction, before the model is loaded
            # and compiled, so the partitioner sees the appended list.
            ops = list(getattr(cc, "splitting_ops", None) or [])
            wanted = ["vllm::moe_forward", "vllm::moe_forward_shared"]
            if self.router_split:
                # the router-split's thin gap op must also be an eager gap
                wanted.append("vllm::sluice_stream_gap")
            added = [op for op in wanted if op not in ops]
            cc.splitting_ops = ops + added
            logger.info(
                "Sluice: piecewise mode — MoE ops added to splitting_ops "
                "(%s); the streaming hook runs in the eager gap.",
                ", ".join(added) if added else "already present",
            )
        if self.router_split and self.piecewise and cc is not None:
            # Dynamo traces the model ONCE, with the profile run's batch
            # (max_num_batched_tokens) as the size hint, and vLLM's custom
            # dispatcher never re-evaluates guards — so a Python size branch
            # in the traced entry is burned in at trace time. The rsplit
            # branch therefore only exists in the artifact if EVERY step fits
            # it: bound the batch so tokens x topk always fits the slots
            # (single-wave contract), and let the trace hint land on the
            # rsplit side.
            topk = 8
            hf = getattr(mc, "hf_config", None) if mc is not None else None
            for src in (hf, getattr(hf, "text_config", None)):
                v = getattr(src, "num_experts_per_tok", None) if src else None
                if v:
                    topk = int(v)
                    break
            thr = max(1, self.expert_cache_slots // topk)
            self._rs_thr = thr
            sched = getattr(cfg, "scheduler_config", None)
            mbt = getattr(sched, "max_num_batched_tokens", None)
            if mbt is not None and mbt > thr:
                raise RuntimeError(
                    "Sluice router-split under piecewise compiles a single "
                    "trace whose size branch is resolved at trace time; the "
                    f"whole batch envelope must fit it. Got "
                    f"max_num_batched_tokens={mbt} > slots//topk={thr}. "
                    f"Set --max-num-batched-tokens {thr} (prefill runs in "
                    f"{thr}-token chunks) or raise SLUICE_SLOTS."
                )
            logger.warning(
                "Sluice: ROUTER-SPLIT envelope — max_num_batched_tokens=%s "
                "<= slots//topk=%d; all steps take the traced split path "
                "(single-wave by construction).",
                mbt,
                thr,
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
        resident_all = num_slots >= local_n
        cache.static_full = resident_all and not self.no_static_full
        # Fully-resident streaming cache: capacity fits every expert but the
        # hook is forced on (SLUICE_NO_STATIC_FULL). Pre-fill all experts so
        # residency starts complete and nothing ever streams — this is the
        # controlled 0-miss operating point for measuring the hook's own cost
        # (legacy vs hook-lite vs lazy-sync) and the lazy-sync ceiling.
        prefill_resident = resident_all and not cache.static_full
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
                if prefill_resident:
                    slot[:local_n].copy_(cpu)
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
        if self.lazy_sync or self.lazy_step:
            # Device mask over GLOBAL ids: True where this rank owns the expert.
            # A lazy-path "miss" is an OWNED expert not resident (map == -1);
            # non-owned ids are legitimately -1 (another rank handles them) and
            # must not count. All-owned at TP=1 (mask all True).
            owned = [0 <= lo < local_n for lo in cache.local_of]
            cache.owned_mask_dev = torch.tensor(
                owned, dtype=torch.bool, device=device
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
        elif prefill_resident:
            # All experts pre-loaded (slot index == local index for the first
            # local_n slots); kept OUT of the SLRU segments so _touch/eviction
            # no-op on them (they never leave — like pins). The hook still runs
            # and the first step establishes the standing map. Extra slots stay
            # free. Residency is complete from step 1, so lazy-sync engages.
            for local in range(local_n):
                cache.slot_of[local] = local
                cache.expert_in_slot[local] = local
            cache.free_slots = list(range(num_slots - 1, local_n - 1, -1))
        # layer.expert_map is a property returning the _expert_map buffer.
        module.register_buffer("_expert_map", cache.expert_map_buf, persistent=False)
        map_host = torch.full((global_n,), -1, dtype=map_dtype, device="cpu")
        if self.pin_memory:
            map_host = map_host.pin_memory()
        cache.map_host = map_host

        if self._pin_layers is not None and not cache.static_full:
            self._apply_pins(cache, len(self._caches))
        if self._counts_dir and not cache.static_full:
            # static_full layers skip the routing hook entirely, so no counts
            # can exist for them; profiling needs slots < local experts.
            self._counts[id(module)] = ([0] * global_n, [0] * global_n)
            self._counts_meta[id(module)] = (
                len(self._caches),
                getattr(module, "layer_name", "") or "",
            )
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

    def _apply_pins(self, cache: _ExpertLayerCache, ordinal: int) -> None:
        """Fill this layer's pinned slots from the pin file (one init-time H2D
        copy per expert). Pins live outside both SLRU segments and the free
        list, so no later path can evict them; their budget is deducted from
        protected_cap so probation keeps its exact unpinned size and prefill
        scan behavior is unchanged."""
        wanted = self._pin_layers.get(ordinal, []) if self._pin_layers else []
        if not wanted:
            return
        budget = cache.protected_cap
        n = 0
        for g in wanted:
            if n >= budget:
                logger.warning(
                    "Sluice: layer %d pin list (%d ids) exceeds the protected "
                    "budget (%d); pinning the first %d only.",
                    ordinal,
                    len(wanted),
                    budget,
                    n,
                )
                break
            if not (0 <= g < len(cache.local_of)):
                continue
            local = cache.local_of[g]
            if local < 0 or local in cache.slot_of or not cache.free_slots:
                continue
            slot = cache.free_slots.pop()
            for name, gpu in cache.gpu_cache.items():
                gpu[slot].copy_(cache.cpu_store[name][local])
            cache.expert_in_slot[slot] = local
            cache.slot_of[local] = slot
            cache.pinned_slots.add(slot)
            n += 1
        if n:
            cache.protected_cap -= n
            self._pin_stats[0] += n
            self._pin_stats[1] += 1

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

        if self.piecewise:
            # Force a dynamo graph break here: the hook (D2H sync + streaming)
            # must run EAGER, outside any captured region. torch.compile then
            # captures the surrounding attention/norms as piecewise graphs.
            apply = torch._dynamo.disable(apply)
        quant_method.apply = apply
        quant_method._sluice_wrapped = True
        if self.gemm_graph and not self._dp_mode:
            self._wrap_mk_gemm_graph(module)

    def _wrap_mk_gemm_graph(self, module: nn.Module) -> None:
        """Wrap the modular impl's ``_fused_experts`` (the pure post-dispatch
        expert GEMM — BELOW even FusedMoEKernel.apply, whose signature still
        carries shared_experts and its aux-stream work) with the
        private-capture replay. Engages only on steps run_moe flagged eligible
        (_gg_step: real, single-wave, decode-sized); else passes through."""
        mk = self._find_mk(module)
        fe = getattr(mk, "_fused_experts", None) if mk is not None else None
        if mk is None or not callable(fe):
            logger.warning(
                "Sluice: SLUICE_GEMM_GRAPH requested but no modular "
                "_fused_experts seam found for this layer; no capture."
            )
            return
        if getattr(mk, "_sluice_gg_wrapped", False):
            return
        offloader = self
        key = id(module)

        def _fused_experts(*fa, **fk):
            if not offloader._gg_step or offloader._gg_disabled:
                return fe(*fa, **fk)
            ntok = None
            for t in list(fa) + list(fk.values()):
                if isinstance(t, torch.Tensor) and t.dim() >= 2:
                    ntok = t.shape[0]
                    break
            if ntok is None or ntok > 16:
                return fe(*fa, **fk)
            out = offloader._gemm_graph_call(key, ntok, fe, fa, fk)
            return out if out is not None else fe(*fa, **fk)

        mk._fused_experts = _fused_experts
        mk._sluice_gg_wrapped = True

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
        if self._counts_dir and pairs and self._split_signal()[2]:
            cnt = self._counts.get(key)
            if cnt is not None:
                # MK path is scan-only policy (no decode signal under DP).
                _dec_arr, scan_arr = cnt
                for g, _local in pairs:
                    scan_arr[g] += 1
                self._counts_tick()
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
            # Wave allowlist (restored after a premature relax). Waves are
            # SEMANTICALLY correct on every kernel class (bit-identical on
            # triton; correct greedy output on marlin under serialized
            # launch), but marlin-class timing hits an async CUDA fault at
            # real speed: single completions pass with the custom fusion
            # passes disabled, yet concurrent load (c4/c16) still crashes —
            # so fusions and CUDA_LAUNCH_BLOCKING merely shift timing; the
            # underlying race (V4-Pro vendor-model path x wave-looped MoE
            # under DP) is unresolved. Default: triton-class kernels wave;
            # others refuse with sizing guidance. SLUICE_MK_WAVES=1 forces
            # (debugging), =0 forbids everywhere.
            allow = os.environ.get("SLUICE_MK_WAVES")
            if allow is None:
                allow = (
                    "1"
                    if "Triton" in self._mk_experts_name.get(key, "")
                    else "0"
                )
            if allow != "1":
                raise RuntimeError(
                    f"Sluice: this step selected {len(pairs)} local experts "
                    f"> {cache.num_slots} slots, but waves on the "
                    f"'{self._mk_experts_name.get(key, 'unknown')}' experts "
                    "kernel hit an unresolved async fault at load (see "
                    "docs). Raise SLUICE_SLOTS to cover the per-step "
                    "working set, cap --max-num-batched-tokens, or force "
                    "with SLUICE_MK_WAVES=1 (unsafe under load)."
                )
            # Rotation window for the remaining waves. No pipelining, so any
            # slots may be reused once the previous wave's kernel is ordered
            # ahead of the fills (see _mk_fill_wave); prefer probation slots
            # to keep bookkeeping simple.
            window = list(cache.probation)
            if not window:
                # (never a pinned slot: pins are immovable by contract)
                window = [
                    s
                    for s in range(cache.num_slots)
                    if s not in cache.pinned_slots
                ][-max(1, cache.num_slots // 4):]
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

    # -- per-expert selection counts (opt-in EPLB-style load profiling) ------

    def _counts_tick(self) -> None:
        """Advance the profiling clock; snapshot every COUNTS_EVERY calls."""
        self._counts_calls += 1
        if self._counts_calls % self._counts_every == 0:
            self._dump_expert_counts(False)

    def _dump_expert_counts(self, final: bool) -> None:
        """Append one cumulative JSONL snapshot per layer (sparse: nonzero
        counts only, keyed by GLOBAL expert id). One file per rank; EP shard
        files merge disjointly. Never raises — profiling must not be able to
        take down serving."""
        if not self._counts_dir or not self._counts:
            return
        try:
            if self._counts_file is None:
                try:
                    import torch.distributed as dist

                    rank = dist.get_rank() if dist.is_initialized() else 0
                except Exception:
                    rank = int(os.environ.get("RANK", "0") or "0")
                os.makedirs(self._counts_dir, exist_ok=True)
                self._counts_file = open(
                    os.path.join(
                        self._counts_dir, f"expert_counts_rank{rank}.jsonl"
                    ),
                    "a",
                )
            self._counts_snap += 1
            for key, (dec_arr, scan_arr) in self._counts.items():
                ord_, name = self._counts_meta.get(key, (-1, ""))
                self._counts_file.write(
                    json.dumps(
                        {
                            "snap": self._counts_snap,
                            "final": final,
                            "ts": time.time(),
                            "hook_calls": self._counts_calls,
                            "steps_small": self._stats["small"].steps,
                            "steps_scan": self._stats["scan"].steps,
                            "layer": ord_,
                            "layer_name": name,
                            "n_global": len(dec_arr),
                            "decode": {
                                str(g): c for g, c in enumerate(dec_arr) if c
                            },
                            "scan": {
                                str(g): c for g, c in enumerate(scan_arr) if c
                            },
                        }
                    )
                    + "\n"
                )
            self._counts_file.flush()
        except Exception:
            logger.exception("Sluice: expert-count dump failed (non-fatal)")

    # -- forward path (fired from the wrapped quant_method.apply) ------------

    def run_moe(self, original_apply, args, kwargs, module, topk_ids):
        """Classic streaming hook, called in place of ``quant_method.apply``.

        Reads the router's ``topk_ids`` (the one host sync per layer),
        classifies the step (decode vs scan), streams the missing experts into
        slots under the SLRU policy, rewrites the expert map, and runs the
        wrapped kernel — once for single-wave steps, once per wave when the
        working set exceeds the cache. Fast paths peel off first: static_full
        layers bypass entirely; lazy-sync/lazy-step skip the host sync;
        hook-lite skips redundant map rewrites on all-hit steps. Under
        ROUTER-SPLIT, decode-sized steps never reach this hook (they run the
        traced entry with the gap op); prefill-sized steps still land here."""
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

        # Lazy-sync fast path: when the standing map is installed AND every
        # local expert is resident (full residency — so a miss is impossible),
        # the kernel has everything it needs on the GPU (standing map +
        # topk_ids). Run it with NO topk_ids D2H sync, no unique/classify/pairs,
        # no map write — this is what eliminates the per-layer host-sync tax.
        # The residency gate makes it exactly correct (0 misses); a device
        # counter proves that. (Partial residency needs a step-level miss
        # fallback, not built here — this measures the ceiling.)
        if (
            self.lazy_sync
            and cache.map_is_standing
            and len(cache.slot_of) >= cache.local_num_experts
        ):
            if not self._lazy_steps and not self._lite_marked:
                self._lite_marked = True
                self._mark_hook_lite_engaged()
            self._lazy_steps += 1
            if self.lazy_miss_check and cache.owned_mask_dev is not None:
                miss = (
                    (cache.expert_map_buf[topk_ids] < 0)
                    & cache.owned_mask_dev[topk_ids]
                ).sum()
                if self._lazy_miss_dev is None:
                    self._lazy_miss_dev = torch.zeros(
                        (), dtype=torch.long, device=topk_ids.device
                    )
                self._lazy_miss_dev += miss
            return original_apply(*args, **kwargs)

        nd, ndec, real_step = self._split_signal()
        # LAZY-STEP fast path: sync-free optimistic layer. The standing map
        # already routes every RESIDENT expert; a device counter accumulates
        # misses for the ONE boundary check in the model-forward wrapper. No
        # D2H, no python planning, no map write. Decode-sized steps only
        # (dim0 is a python shape read — free); big scans keep the classic
        # path so prefill never pays 2× reruns.
        if (
            self._lazy_now
            and real_step
            and cache.map_is_standing
            and cache.owned_mask_dev is not None
            and topk_ids.shape[0] <= 16
        ):
            miss = (
                (cache.expert_map_buf[topk_ids] < 0)
                & cache.owned_mask_dev[topk_ids]
            ).sum()
            if self._lazy_miss_dev is None:
                self._lazy_miss_dev = torch.zeros(
                    (), dtype=torch.long, device=topk_ids.device
                )
            self._lazy_miss_dev += miss
            self._lazy_ran = True
            return original_apply(*args, **kwargs)

        # One host sync per layer: the router's decisions gate everything.
        # A single D2H copy, then unique on CPU (slicing decode vs prefill
        # rows costs no extra sync).
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
            # Decode-set capacity for CLASSIFICATION is pinned + dynamic
            # protected (protected_cap alone shrinks when pins deduct their
            # budget, which would silently reclassify decode-ish steps as
            # scans and disable their promotions — the Phase-1 arms artifact).
            decode_cap = cache.protected_cap + len(cache.pinned_slots)
            heuristic_friendly = real_step and ws <= max(decode_cap, 1)
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
            if self._counts_dir:
                cnt = self._counts.get(id(module))
                if cnt is not None:
                    dec_arr, scan_arr = cnt
                    for g, _local, dcls in pairs:
                        (dec_arr if dcls else scan_arr)[g] += 1
                    self._counts_tick()

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
            # real_step only: vLLM's memory-profiling and piecewise-capture
            # passes are dummy runs — staging/capturing our private graph
            # during THEIR capture corrupts memory (observed: illegal access).
            gg_ntok = num_tokens if (
                self.gemm_graph
                and not self._gg_disabled
                and real_step
                and num_tokens <= 16
                and not (
                    torch.cuda.is_available()
                    and torch.cuda.is_current_stream_capturing()
                )
            ) else 0
            if self.hook_lite:
                # Standing map covers every resident expert, so the selected
                # (all-hit) set is already routed. Rewrite only when residency
                # changed (taken) or the map isn't standing yet; otherwise skip
                # the H2D and its map_ev stall entirely.
                if taken or not cache.map_is_standing:
                    self._write_standing_map(cache)
                    self._lite_writes += 1
                else:
                    self._lite_skips += 1
                    if not self._lite_marked:
                        self._lite_marked = True
                        self._mark_hook_lite_engaged()
            else:
                self._write_map(cache, wave0, pinned=True)
            if taken:  # streamed something: kernel must wait for the copies
                self._compute_wait_copies(copy_stream)
            if gg_ntok:
                # Eligible step: the wrapped modular kernel (mk.forward, BELOW
                # apply and its shared-experts/aux-stream work) captures or
                # replays the routed GEMM. Flag is step-scoped.
                self._gg_step = True
                try:
                    return original_apply(*args, **kwargs)
                finally:
                    self._gg_step = False
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
            # (never a pinned slot: pins are immovable by contract)
            rot = [
                s for s in range(cache.num_slots) if s not in cache.pinned_slots
            ][-max(1, cache.num_slots // 4):]
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
        # Any raw (sparse/wave) map write invalidates the standing invariant;
        # _write_standing_map re-establishes it after this returns.
        cache.map_is_standing = False

    def _write_standing_map(self, cache: _ExpertLayerCache) -> None:
        """Install the STANDING map: every currently-resident expert points at
        its slot (-1 elsewhere). Correct for any all-hit step regardless of
        which experts it selects, so subsequent hits need no rewrite. Costs
        one map H2D — paid only when residency changed."""
        wave = [
            (cache.global_of_local[local], slot)
            for slot, local in enumerate(cache.expert_in_slot)
            if local is not None
        ]
        self._write_map(cache, wave, pinned=True)
        cache.map_is_standing = True

    def _gemm_graph_call(self, key, ntok, original_apply, args, kwargs):
        """Replay (capturing on first sight) a private CUDA graph of this
        layer's fused-experts call for this token count.

        A graph bakes POINTERS: per-step tensors (dim0 == ntok: hidden states,
        topk ids/weights, router logits) are staged into static buffers and
        copied in before each replay; persistent tensors (weights, scales —
        stable layer attributes) and the expert-map buffer are captured via
        their live pointers, whose CONTENTS the gap may update between
        replays. Only single-wave decode-sized steps reach here; anything
        unexpected disables the feature for the process and falls back to the
        eager call (returns None). Correct by construction: the replayed work
        is exactly the captured original_apply with identical inputs."""
        try:
            bucket = (key, ntok)
            entry = self._gg.get(bucket)
            if entry is None:
                # First sighting: record every tensor arg's object identity,
                # run eager. On the second sighting, args whose identity
                # CHANGED are per-step (must be staged into static buffers);
                # args with the SAME tensor object (layer weights, the expert
                # map — module attributes) are persistent and captured via
                # their live pointers. Identity-based classification has no
                # holes, unlike any shape heuristic: ANY per-step tensor left
                # unstaged would be captured as a dangling pointer and replay
                # garbage once its memory is recycled.
                self._gg[bucket] = {
                    "ids": (
                        [id(a) if isinstance(a, torch.Tensor) else None
                         for a in args],
                        {k: id(v) if isinstance(v, torch.Tensor) else None
                         for k, v in kwargs.items()},
                    )
                }
                return None  # eager this step
            if "graph" not in entry and "ids" in entry:
                aids, kids = entry["ids"]
                if len(aids) != len(args) or set(kids) != set(kwargs):
                    raise RuntimeError("apply arg structure changed")

                forced = self._gg_force_perstep.get(key, set())

                def stage(obj, prev_id, tag):
                    if isinstance(obj, torch.Tensor):
                        if id(obj) == prev_id and tag not in forced:
                            return obj, False  # persistent: same object
                        return obj.detach().clone(), True  # per-step: stage
                    return obj, False

                sargs, aflag = [], []
                for i, (a, pid) in enumerate(zip(args, aids)):
                    s, f = stage(a, pid, ("a", i))
                    sargs.append(s)
                    aflag.append(f)
                skw, kflag = {}, {}
                for k, v in kwargs.items():
                    s, f = stage(v, kids[k], ("k", k))
                    skw[k] = s
                    kflag[k] = f
                sargs = tuple(sargs)
                if os.environ.get("SLUICE_GG_DEBUG") == "1":
                    inv = []
                    for i, a in enumerate(args):
                        inv.append(
                            f"arg{i}:{type(a).__name__}"
                            + (f"{tuple(a.shape)}{'*' if aflag[i] else ''}"
                               if isinstance(a, torch.Tensor) else "")
                        )
                    for k, v in kwargs.items():
                        inv.append(
                            f"{k}:{type(v).__name__}"
                            + (f"{tuple(v.shape)}{'*' if kflag[k] else ''}"
                               if isinstance(v, torch.Tensor) else "")
                        )
                    logger.warning(
                        "Sluice GG_DEBUG bucket(ntok=%d): %s (*=per-step)",
                        ntok, " ".join(inv),
                    )

                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    original_apply(*sargs, **skw)  # allocator warmup
                torch.cuda.current_stream().wait_stream(side)
                # The eager call mutates hidden_states in place (vllm registers
                # moe_forward with mutates_args) — restore pristine inputs so
                # the capture records the intended computation.
                for live, static, f in zip(args, sargs, aflag):
                    if f:
                        static.copy_(live)
                for k, f in kflag.items():
                    if f:
                        skw[k].copy_(kwargs[k])

                graph = torch.cuda.CUDAGraph()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    with torch.cuda.graph(graph):
                        out = original_apply(*sargs, **skw)
                torch.cuda.current_stream().wait_stream(side)
                if not isinstance(out, torch.Tensor):
                    raise TypeError(f"apply returned {type(out)}")
                entry = {
                    "graph": graph,
                    "sargs": sargs,
                    "aflag": aflag,
                    "skw": skw,
                    "kflag": kflag,
                    "out": out,
                }
                self._gg[(key, ntok)] = entry
                self._gg_captures += 1

            # copy this step's per-step tensors into the captured buffers; for
            # persistent args VERIFY identity still holds (a replaced buffer
            # would be a stale captured pointer → must disable, not corrupt)
            if len(args) != len(entry["sargs"]):
                raise RuntimeError("apply arg count changed")

            def heal(tag):
                # Identity lied (buffer reused for the first sightings, then
                # swapped — e.g. output_alias). Force it per-step and rebuild
                # every bucket of this layer; this step runs eager.
                self._gg_force_perstep.setdefault(key, set()).add(tag)
                for b in [b for b in self._gg if b[0] == key]:
                    del self._gg[b]
                logger.warning(
                    "Sluice: GEMM-graph reclassified %s as per-step for layer "
                    "%d (identity changed); recapturing.", tag, key,
                )

            for i, (live, static, f) in enumerate(
                zip(args, entry["sargs"], entry["aflag"])
            ):
                if f:
                    static.copy_(live, non_blocking=True)
                elif isinstance(live, torch.Tensor) and live is not static:
                    heal(("a", i))
                    return None
            for k, f in entry["kflag"].items():
                if f:
                    entry["skw"][k].copy_(kwargs[k], non_blocking=True)
                elif (
                    isinstance(kwargs[k], torch.Tensor)
                    and kwargs[k] is not entry["skw"][k]
                ):
                    heal(("k", k))
                    return None
            entry["graph"].replay()
            self._gg_replays += 1
            # Mirror the eager call's side effects: the op mutates
            # hidden_states in place and downstream may read the LIVE tensor —
            # propagate every per-step buffer back (harmless for unmutated
            # ones; restores the contract for mutated ones).
            for live, static, f in zip(args, entry["sargs"], entry["aflag"]):
                if f:
                    live.copy_(static, non_blocking=True)
            for k, f in entry["kflag"].items():
                if f:
                    kwargs[k].copy_(entry["skw"][k], non_blocking=True)
            # clone: downstream must not alias the static output buffer the
            # next replay will overwrite.
            return entry["out"].clone()
        except Exception:
            if not self._gg_disabled:
                self._gg_disabled = True
                logger.exception(
                    "Sluice: GEMM-graph disabled after error (eager fallback)."
                )
            return None

    def _register_stream_gap_op(self) -> None:
        """Register ``vllm::sluice_stream_gap`` — the router-split's thin
        eager gap. Functionally pure to the graph (returns a clone consumed by
        fused_experts, enforcing ordering by dataflow); its side effect is
        streaming this step's missing experts into slots and refreshing the
        standing map, which the captured fused_experts then reads by pointer.

        Under SLUICE_RS_STAGED (kept negative result) this also registers
        ``vllm::sluice_stage_ids``: a capturable async D2H of topk_ids into a
        pinned per-layer buffer plus a doorbell stamp, so the gap can
        spin-wait the doorbell instead of syncing the stream."""
        from vllm.utils.torch_utils import direct_register_custom_op

        offloader = self

        def _sluice_stream_gap(
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            layer_idx: int,
        ) -> torch.Tensor:
            offloader._router_split_gap(int(layer_idx), topk_ids)
            return hidden_states.clone()

        def _sluice_stream_gap_fake(
            hidden_states: torch.Tensor,
            topk_ids: torch.Tensor,
            layer_idx: int,
        ) -> torch.Tensor:
            return torch.empty_like(hidden_states)

        direct_register_custom_op(
            op_name="sluice_stream_gap",
            op_func=_sluice_stream_gap,
            mutates_args=[],
            fake_impl=_sluice_stream_gap_fake,
        )

        if self._rs_staged:

            def _sluice_stage_ids(
                topk_ids: torch.Tensor, layer_idx: int
            ) -> torch.Tensor:
                st = offloader._rs_stage[int(layer_idx)]
                n = topk_ids.numel()
                # Ordered on the compute stream: ids land in pinned memory,
                # THEN the doorbell advances — stamp visible implies ids
                # visible. Both copies are capturable (async memcpy nodes).
                st["pin_ids"][:n].copy_(
                    topk_ids.view(-1).to(torch.int32), non_blocking=True
                )
                st["dev_stamp"] += 1
                st["pin_stamp"].copy_(st["dev_stamp"], non_blocking=True)
                return topk_ids

            def _sluice_stage_ids_fake(
                topk_ids: torch.Tensor, layer_idx: int
            ) -> torch.Tensor:
                return torch.empty_like(topk_ids)

            direct_register_custom_op(
                op_name="sluice_stage_ids",
                op_func=_sluice_stage_ids,
                mutates_args=[],
                fake_impl=_sluice_stage_ids_fake,
            )

    def _router_split_gap(self, idx: int, topk_ids: torch.Tensor) -> None:
        """Body of ``vllm::sluice_stream_gap`` — router-split's only eager
        region: one D2H of the tiny topk tensor (or a doorbell read under
        SLUICE_RS_STAGED), stream the misses, refresh the standing map the
        captured fused_experts reads by pointer. SLUICE_RS_FAST_HIT skips the
        LRU bookkeeping on all-hit steps (exact — the D2H anchor stays).
        Single-wave by contract (decode-sized steps only, slots >= working
        set); overflow is a loud error, never a wrong answer."""
        cache = self._rs_caches[idx]
        self._rs_gaps += 1
        # Self-report: when rsplit covers every small batch the classic hook
        # (and its periodic DIAG printer) never runs — engagement of BOTH the
        # hit and miss paths must be visible from the gap itself.
        if self._rs_gaps == 1 or self._rs_gaps % 2000 == 0:
            logger.warning(
                "Sluice DIAG[rs]: router-split layers=%d gap-calls=%d "
                "misses=%d fast-hits=%d stage-falls=%d%s",
                len(self._rs_caches),
                self._rs_gaps,
                self._rs_misses,
                self._rs_fast_hits,
                self._rs_stage_falls,
                " NOOP-GAP(OUTPUTS INVALID)" if self._rs_noop else "",
            )
        if self._rs_noop:
            return
        ids_list = None
        if self._rs_staged and idx < len(self._rs_stage):
            st = self._rs_stage[idx]
            target = st["seen"] + 1
            pin_stamp = st["pin_stamp"]
            t0 = time.perf_counter()
            while int(pin_stamp[0]) < target:
                if time.perf_counter() - t0 > 0.01:
                    break
            if int(pin_stamp[0]) >= target:
                st["seen"] = int(pin_stamp[0])
                ids_list = st["pin_ids"][: topk_ids.numel()].tolist()
            else:
                # Capture pass (recorded, not executed) or a stall: classic
                # sync path, then resync the doorbell bookkeeping.
                self._rs_stage_falls += 1
                st["seen"] = int(pin_stamp[0])
        if ids_list is None:
            ids_list = topk_ids.to("cpu").view(-1).tolist()
        local_of = cache.local_of
        n_global = len(local_of)
        if self._rs_fast_hit and cache.map_is_standing:
            # All-hit fast path: the standing map already covers every
            # resident expert, so a step with no misses needs NO unique, NO
            # LRU touches, NO map-write check. Recency goes stale on hit
            # steps — that only shifts victim choice on (rare) misses, never
            # correctness. The D2H (sync or staged doorbell) stays: it is
            # the exactness anchor.
            slot_of = cache.slot_of
            for g in ids_list:
                if 0 <= g < n_global:
                    local = local_of[g]
                    if local >= 0 and local not in slot_of:
                        break
            else:
                self._rs_fast_hits += 1
                return
        # sorted(set()) matches torch.unique's sorted order exactly, so slot
        # placement sequences stay comparable across variants.
        selected = sorted(set(ids_list))
        missing = []
        hit_slots = set()
        changed = False
        for g in selected:
            if 0 <= g < n_global:
                local = local_of[g]
                if local < 0:
                    continue
                slot = cache.slot_of.get(local)
                if slot is None:
                    missing.append(local)
                else:
                    hit_slots.add(slot)
                    self._touch(cache, slot, True)
        if missing:
            copy_stream = self._get_copy_stream(cache.device)
            if copy_stream is not None:
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream())
                copy_stream.wait_event(ev)
            # Seed the no-evict set with THIS STEP'S HIT SLOTS: evicting a
            # selected resident to admit a selected miss hands the GEMM a
            # slot whose weights were overwritten mid-step (garbage/NaN at
            # uniques == slots). The envelope guarantees feasibility:
            # |selected| <= slots  =>  non-selected slots >= misses.
            claimed = set(hit_slots)
            for local in missing:
                slot = self._acquire_slot(cache, local, claimed, True)
                if slot is None:
                    raise RuntimeError(
                        "Sluice router-split: working set exceeds slots on a "
                        "decode-sized step; raise SLUICE_SLOTS."
                    )
                self._stream_in(cache, local, slot, copy_stream)
                claimed.add(slot)
            self._rs_misses += len(missing)
            self._compute_wait_copies(copy_stream)
            changed = True
        if changed or not cache.map_is_standing:
            self._write_standing_map(cache)

    def attach_router_split(self, model: nn.Module) -> None:
        """Arm ROUTER-SPLIT: patch each supported MoE layer's
        ``runner._forward_entry`` with the traced split path

            gate linear + select_experts   [captured]
            -> sluice_stream_gap           [eager: sync + stream + map]
            -> torch.ops.vllm.fused_experts [captured; weights/map by pointer]
            -> shared experts              [captured]

        for steps of <= threshold tokens (slots//topk — single-wave by
        construction); larger steps call the original entry, i.e. the stock
        opaque path with the classic wave-capable hook. Layers missing a
        required seam (no runner, monolithic kernel, fused gate, naive
        dispatch, pcp>1, static_full) keep the stock path untouched — fail
        closed to correctness, counted in the ``skips`` log; the eager
        bit-gate validates the rest."""
        if not self.router_split:
            return
        patched = 0
        skips: dict = {}

        def skip(reason):
            skips[reason] = skips.get(reason, 0) + 1

        # vLLM <= 0.23: the cache module is the FusedMoE layer and holds its
        # MoERunner as ``.runner``. vLLM >= 0.25: FusedMoE is a factory, the
        # cache module is the weight-owning RoutedExperts, and the backlink is
        # inverted — the runner holds ``.routed_experts``. Build the reverse
        # map once so both shapes resolve; a cache module matching neither
        # falls into the existing no-runner skip (fail closed).
        runner_by_cache = {
            id(m.routed_experts): m
            for m in model.modules()
            if getattr(m, "routed_experts", None) is not None
            and hasattr(m, "_forward_entry")
        }

        seen_moe = 0
        for module in model.modules():
            cache = self._caches.get(id(module))
            if cache is None:
                continue
            seen_moe += 1
            runner = getattr(module, "runner", None)
            if runner is None:
                runner = runner_by_cache.get(id(module))
            if runner is None:
                skip("no-runner")
                continue
            if cache.static_full:
                skip("static-full")
                continue
            try:
                qm = runner._quant_method
                if getattr(qm, "is_monolithic", False):
                    skip("monolithic")
                    continue
                # Runner-held gate is fine — the traced entry mirrors stock
                # (`router_logits, _ = self.gate(hidden)`), a plain traceable
                # linear. Only the fused-gate variant is out of scope.
                gate = getattr(runner, "gate", None)
                if gate is not None and getattr(runner, "_fse_fuse_gate", False):
                    skip("fused-gate")
                    continue
                if getattr(runner, "do_naive_dispatch_combine", False):
                    skip("naive-dispatch")
                    continue
                mc = getattr(runner, "moe_config", None)
                if mc is not None and getattr(mc, "pcp_size", 1) > 1:
                    skip("pcp")
                    continue
                w13 = module.w13_weight
                w2 = module.w2_weight
                gne = int(module.global_num_experts)
                emap = cache.expert_map_buf
                se = getattr(runner, "_shared_experts", None)
                shared_layer = getattr(se, "_layer", None) if se is not None else None
                sel = runner.router.select_experts
                orig_entry = runner._forward_entry
                topk = getattr(mc, "experts_per_token", None) if mc else None
                if topk is None:
                    topk = getattr(module, "top_k", None) or 8
            except AttributeError as e:
                skip(f"attr:{e}")
                continue
            idx = len(self._rs_caches)
            self._rs_caches.append(cache)
            # Capture-safe smallness bound: the rsplit path is single-wave by
            # contract, so worst-case uniques (tokens x topk) must fit the
            # slots. Under piecewise the config-time envelope pins this (and
            # bounds the whole batch); in eager the branch is evaluated per
            # step and larger steps fall back to the wave-capable classic
            # hook.
            thr = self._rs_thr
            if thr is None:
                thr = max(1, min(16, self.expert_cache_slots // int(topk)))
            if self._rs_staged:
                dev = module.w13_weight.device
                self._rs_stage.append(
                    {
                        "pin_ids": torch.empty(
                            (thr * int(topk),),
                            dtype=torch.int32,
                            pin_memory=True,
                        ),
                        "pin_stamp": torch.zeros(
                            (1,), dtype=torch.int32, pin_memory=True
                        ),
                        "dev_stamp": torch.zeros(
                            (1,), dtype=torch.int32, device=dev
                        ),
                        "seen": 0,
                    }
                )

            def make_entry(
                idx, w13, w2, gne, emap, shared_layer, sel, orig, gate, thr,
                staged,
            ):
                """Bind THIS layer's tensors/callables into a fresh traced
                entry (a factory, so the loop can't rebind the closure to the
                last layer's variables). ``emap`` and the weights are captured
                BY POINTER; the gap op rewrites their contents per step."""

                def _forward_entry(hs, rl, sei, iid, lname, unpad):
                    if hs.shape[0] > thr:
                        return orig(hs, rl, sei, iid, lname, unpad)
                    if gate is not None:
                        rl, _ = gate(hs)  # mirrors stock _forward_impl
                    tw, ti = sel(
                        hidden_states=hs, router_logits=rl, input_ids=iid
                    )
                    if staged:
                        # Captured async D2H + doorbell, inside the piece.
                        ti = torch.ops.vllm.sluice_stage_ids(ti, idx)
                    h = torch.ops.vllm.sluice_stream_gap(hs, ti, idx)
                    routed = torch.ops.vllm.fused_experts(
                        h,
                        w13,
                        w2,
                        tw,
                        ti,
                        activation="silu",
                        global_num_experts=gne,
                        expert_map=emap,
                    )
                    if shared_layer is None:
                        return routed
                    sh = shared_layer(sei if sei is not None else hs)
                    return (sh, routed)

                return _forward_entry

            if idx == 0 and os.environ.get("SLUICE_RS_DUMP", "0") == "1":
                # One-shot path forensics: what kernel stack does the classic
                # apply route through on THIS model (vs our direct
                # torch.ops.vllm.fused_experts call)?
                mk = getattr(qm, "fused_experts", None)
                logger.warning(
                    "RS-DUMP: qm=%s mk=%s pf=%s expert_impl=%s gne=%d "
                    "moe_config=%.400s",
                    type(qm).__name__,
                    type(mk).__name__ if mk is not None else None,
                    type(getattr(mk, "prepare_finalize", None)).__name__,
                    type(getattr(mk, "fused_experts", None)).__name__,
                    gne,
                    repr(mc),
                )
            runner._forward_entry = make_entry(
                idx, w13, w2, gne, emap, shared_layer, sel, orig_entry, gate,
                thr, self._rs_staged,
            )
            patched += 1
        self._rs_layers = patched
        logger.warning(
            "Sluice: ROUTER-SPLIT armed on %d/%d MoE layers (traced "
            "select_experts + captured fused_experts; gap = stream+map only)."
            " skips=%s",
            patched,
            seen_moe,
            skips or "{}",
        )

    def attach_model(self, model: nn.Module) -> None:
        """Model-handle attach point, called by the plugin after load_model.

        Always arms ROUTER-SPLIT first (a no-op unless SLUICE_ROUTER_SPLIT=1;
        see ``attach_router_split``). Then, only under SLUICE_LAZY_STEP (kept
        negative result), wraps the model's forward so each engine step runs
        the offloader in sync-free mode with ONE boundary check: zero misses
        → commit (the per-layer syncs collapse to one); misses → re-run the
        same forward with the classic hooked path, which streams the misses
        and refreshes the standing maps — exact because the same-step rerun
        overwrites the same KV slots with corrected values and sampling
        happens once, afterwards."""
        self.attach_router_split(model)
        if not self.lazy_step or getattr(model, "_sluice_lazy_wrapped", False):
            return
        orig_forward = model.forward
        offloader = self

        def forward(*a, **k):
            if offloader._lazy_disabled or not offloader._caches:
                return orig_forward(*a, **k)
            try:
                if offloader._lazy_miss_dev is not None:
                    offloader._lazy_miss_dev.zero_()
                offloader._lazy_ran = False
                offloader._lazy_now = True
                try:
                    out = orig_forward(*a, **k)
                finally:
                    offloader._lazy_now = False
                if offloader._lazy_ran and offloader._lazy_miss_dev is not None:
                    if int(offloader._lazy_miss_dev.item()):  # the ONE sync
                        offloader._lstep_stats[1] += 1
                        out = orig_forward(*a, **k)  # classic, streams misses
                    else:
                        offloader._lstep_stats[0] += 1
                else:
                    offloader._lstep_stats[2] += 1
                return out
            except Exception:
                offloader._lazy_disabled = True
                logger.exception(
                    "Sluice: LAZY-STEP disabled after error (classic fallback)."
                )
                return orig_forward(*a, **k)

        model.forward = forward
        model._sluice_lazy_wrapped = True
        logger.warning(
            "Sluice: LAZY-STEP armed — one boundary sync per forward, classic "
            "re-run on miss."
        )

    def _mark_hook_lite_engaged(self) -> None:
        """Filesystem proof the standing-map fast path actually executed (INFO
        logs are invisible in engine subprocesses; a null tok/s result must be
        distinguishable from 'never ran'). Best-effort; never fatal."""
        path = os.environ.get("SLUICE_HOOK_LITE_MARKER")
        if not path:
            return
        try:
            try:
                import torch.distributed as dist

                rank = dist.get_rank() if dist.is_initialized() else 0
            except Exception:
                rank = int(os.environ.get("RANK", "0") or "0")
            with open(f"{path}.rank{rank}", "w") as f:
                f.write("engaged\n")
        except Exception:
            logger.exception("Sluice: hook-lite marker write failed")

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
            if self.hook_lite:
                tot = self._lite_skips + self._lite_writes
                logger.warning(
                    "Sluice DIAG[lite]: map-writes skipped=%d written=%d "
                    "(%.1f%% of single-wave steps skipped)",
                    self._lite_skips,
                    self._lite_writes,
                    100.0 * self._lite_skips / max(tot, 1),
                )
            if self.gemm_graph:
                logger.warning(
                    "Sluice DIAG[gg]: gemm-graph replays=%d captures=%d "
                    "disabled=%s",
                    self._gg_replays,
                    self._gg_captures,
                    self._gg_disabled,
                )
            if self.router_split:
                logger.warning(
                    "Sluice DIAG[rs]: router-split layers=%d gap-calls=%d",
                    self._rs_layers,
                    self._rs_gaps,
                )
            if self.lazy_step:
                logger.warning(
                    "Sluice DIAG[lstep]: clean=%d miss-rerun=%d classic=%d "
                    "disabled=%s",
                    self._lstep_stats[0],
                    self._lstep_stats[1],
                    self._lstep_stats[2],
                    self._lazy_disabled,
                )
            if self.lazy_sync:
                # One .item() here (periodic, off the fast path) — the sync-free
                # invariant is that misses==0; anything else means the residency
                # gate was violated and the run is NOT correct.
                misses = (
                    int(self._lazy_miss_dev.item())
                    if self._lazy_miss_dev is not None
                    else 0
                )
                logger.warning(
                    "Sluice DIAG[lazy]: sync-free layer-calls=%d misses=%d "
                    "%s",
                    self._lazy_steps,
                    misses,
                    "(EXACT)" if misses == 0 else "(!! INCORRECT: gate violated)",
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
