# SPDX-License-Identifier: Apache-2.0
"""Generate Sluice's result charts (matplotlib).

Measured (4xH100): V4-Pro GPU weights at load 9.49 GiB; KV 24.4 GiB @ util 0.45;
16 of 96 local experts resident. V2-Lite (BF16, 1 GPU) ~31 GiB, fits natively.
Estimated from the 805 GiB checkpoint: V4-Pro per-rank expert shard ~191 GiB,
16-slot cache ~32 GiB, ~2 GiB per expert.

    python assets/make_charts.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

TEAL, INDIGO, AMBER, GREY, INK = "#0DB8AB", "#4338CA", "#F5A926", "#C9CED6", "#1F2430"
RED, GREEN = "#B0152F", "#0B7A33"
H100 = 80

# Cohesive, modern look applied to every figure (call _style() before plotting).
INK_SOFT, MUTED, HAIR = "#2B313B", "#6B7480", "#E7EBEF"


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],  # clean + ships every glyph we use
        "font.size": 12,
        "text.color": INK,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 14,
        "axes.titlecolor": INK,
        "axes.labelsize": 11, "axes.labelcolor": INK_SOFT, "axes.labelweight": "medium",
        "axes.edgecolor": HAIR, "axes.linewidth": 1.2,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": HAIR, "grid.linewidth": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelsize": 10.5, "ytick.labelsize": 10.5,
        "figure.dpi": 200, "savefig.dpi": 200,
    })


def _retain_cmap():
    """White → teal sequential: more throughput retained reads greener."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "retain", ["#FBFDFD", "#CFEFEB", "#7FD8CE", "#28BBA9", "#0E9C8C"])


def comparison_chart(path):
    """Per-GPU memory and — the headline — what's LEFT for KV cache up to the 80 GiB
    H100 line. V2-Lite fits a GPU outright; V4-Pro's experts don't (~101 GiB/GPU →
    OOM), but routing-aware offload keeps a small slot cache resident instead of the
    full shard, turning OOM into KV headroom — down to a single H100. 8×H100 weights
    & KV are measured; expert/slot-cache sizes (~2 GiB/expert) are estimated from the
    805 GiB checkpoint. 'Room for KV' = 80 − (weights + slot cache)."""
    TOP = 92
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=200)
    BARW = 0.6

    # per-GPU GiB. kv = 80 - (ne + cache) is the space the bar leaves for KV cache.
    bars = [
        dict(x=0, lab="V2-Lite\n1×H100",            ne=31.0, cache=0,  exp=0,    oom=False),
        dict(x=1, lab="V2-Lite\n1×H100 + Sluice",   ne=2.6,  cache=26, exp=0,    oom=False,
             note="cache ≈ all\nexperts"),
        dict(x=2, lab="V4-Pro\nno offload",         ne=6.5,  cache=0,  exp=94.0, oom=True),
        dict(x=3, lab="V4-Pro\n8×H100 + Sluice",    ne=6.54, cache=32, exp=0,    oom=False,
             note="16 of 384\nresident"),
        dict(x=4, lab="V4-Pro\n1×H100 + Sluice",    ne=41.0, cache=16, exp=0,    oom=False,
             note="8 of 384\nresident"),
    ]

    def break_mark(xc, y):
        ax.add_patch(Rectangle((xc - BARW / 2, y - 1.3), BARW, 2.6, facecolor="white",
                     edgecolor="none", zorder=6, clip_on=False))
        for off in (-1.0, 1.0):
            ax.plot([xc - BARW / 2, xc + BARW / 2], [y + off - 1.0, y + off + 1.0],
                    color="#9AA1AC", lw=1.3, zorder=7, clip_on=False, solid_capstyle="round")

    for b in bars:
        x = b["x"]
        ax.bar(x, b["ne"], color=TEAL, width=BARW, zorder=3)
        top = b["ne"]
        if b["oom"]:
            # experts kept resident overrun the GPU -> draw to the top edge & break it
            ax.bar(x, TOP - top, bottom=top, color=GREY, width=BARW, zorder=3)
            break_mark(x, TOP - 5)
            ax.text(x, TOP - 11, "experts\ndon't fit", ha="center", va="center",
                    color="white", fontsize=10.5, fontweight="bold", zorder=8)
            ax.annotate("✗ OOM\n~101 GiB/GPU", (x, H100 - 1), ha="center", va="top",
                        color=RED, fontsize=10.5, fontweight="bold", zorder=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=RED, lw=1.2))
            continue
        if b["cache"]:
            ax.bar(x, b["cache"], bottom=top, color=INDIGO, width=BARW, zorder=3)
            if b.get("note") and b["cache"] >= 14:
                ax.text(x, top + b["cache"] / 2, b["note"], ha="center", va="center",
                        color="white", fontsize=8, fontweight="medium", zorder=5,
                        linespacing=1.2)
            top += b["cache"]
        kv = H100 - top
        ax.bar(x, kv, bottom=top, color=AMBER, width=BARW, zorder=3, alpha=0.92)
        ax.text(x, top + kv / 2, f"✓ {kv:.0f} GiB\nfor KV", ha="center", va="center",
                color="#7A4E00", fontsize=10, fontweight="bold", zorder=5)

    # group context: the two model totals, with a soft divider between them
    ax.axvline(1.5, color=HAIR, lw=1.3, ls=(0, (4, 4)), zorder=1)
    ax.text(0.5, TOP - 1.0, "V2-Lite · 29 GB", ha="center",
            va="top", color=MUTED, fontsize=10, style="italic")
    ax.text(3.3, TOP - 1.0, "V4-Pro · 805 GB",
            ha="center", va="top", color=MUTED, fontsize=10, style="italic")

    ax.axhline(H100, ls="--", lw=1.7, color="#E0457B", zorder=2)
    ax.text(4.52, H100 + 0.6, "H100 = 80 GiB", color="#E0457B", ha="right",
            va="bottom", fontsize=9.5, fontweight="bold")

    ax.set_xticks([b["x"] for b in bars])
    ax.set_xticklabels([b["lab"] for b in bars], fontsize=9.5)
    ax.set_ylabel("per-GPU memory (GiB)", fontsize=10.5)
    ax.set_ylim(0, TOP)
    ax.set_xlim(-0.6, 4.6)
    ax.set_title("After the weights, what's left for KV cache",
                 fontsize=13.5, fontweight="bold", color=INK)
    ax.legend(handles=[
        Patch(color=TEAL, label="non-expert weights"),
        Patch(color=INDIGO, label="experts kept resident — Sluice slot cache (est.)"),
        Patch(color=GREY, label="experts kept resident — full shard (no offload)"),
        Patch(facecolor=AMBER, alpha=0.92, label="room left for KV cache (to 80 GiB)"),
    ], fontsize=8.5, loc="upper center", frameon=False, ncol=1,
        bbox_to_anchor=(0.5, -0.16))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.115, "Per-GPU footprint. V2-Lite ~31 GiB resident (measured); "
             "V4-Pro 8×H100 weights 6.5 GiB & KV measured, expert/cache sizes (~2 GiB/"
             "expert) estimated from the 805 GiB checkpoint. V4-Pro 1×H100 fits in "
             "memory; single-GPU serving not yet confirmed (2×H100 is the smallest "
             "confirmed).", ha="center", fontsize=7.3, color="#7A828F", wrap=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def residency_chart(path):
    cols, rows, resident = 16, 6, 16
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=200)
    n = 0
    for r in range(rows):
        for c in range(cols):
            ax.add_patch(Rectangle((c, rows - 1 - r), 0.86, 0.86,
                         facecolor=AMBER if n < resident else GREY,
                         edgecolor="white", lw=1.2))
            n += 1
    ax.set_xlim(-0.3, cols + 0.3)
    ax.set_ylim(-0.3, rows + 0.3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("V4-Pro working set: 16 of 96 experts resident per layer/rank",
                 fontsize=12, fontweight="bold", color=INK, pad=10)
    fig.text(0.5, 0.02, "amber = streamed into GPU slots by routing   ·   "
             "grey = held in host RAM", ha="center", fontsize=9, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def tradeoff_chart(path):
    per_expert = 191 / 96  # GiB
    slots = [8, 16, 24, 32, 48, 64, 96]
    cache = [s * per_expert for s in slots]
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=200)
    ax.plot(slots, cache, "-o", color=INDIGO, lw=2.2, label="resident cache (GiB)")
    ax.axhline(46, ls="--", lw=1.5, color="#E0457B")
    ax.text(95, 48, "VRAM left for experts after weights+KV (~46 GiB)",
            ha="right", color="#E0457B", fontsize=8.5)
    ax.scatter([16], [32], s=120, color=AMBER, zorder=5, edgecolor="white")
    ax.annotate("validated: 16 slots ≈ 32 GiB", (16, 32), (24, 70),
                arrowprops=dict(arrowstyle="->", color=INK), fontsize=9, color=INK)
    ax.set_xlabel("SLUICE_SLOTS (resident experts per layer/rank)", fontsize=10)
    ax.set_ylabel("GPU cache size (GiB)", fontsize=10)
    ax.set_title("Tuning the cache (V4-Pro, EP=4 — projected)",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(fontsize=9, frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "~2 GiB per expert; raise slots until a step's experts "
             "fit, lower to save VRAM", ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def throughput_chart(path):
    # Measured: 8 sequences x 128 tokens, greedy, enforce_eager.
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(8.6, 4.2), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]}
    )
    axL.bar([0, 1], [229.3, 195.9], color=[GREY, INDIGO], width=0.6)
    axL.set_xticks([0, 1])
    axL.set_xticklabels(["vanilla\n(resident)", "+ Sluice\n(slots=64)"], fontsize=10)
    axL.set_ylabel("decode throughput (tok/s)", fontsize=10)
    axL.set_title("V2-Lite · 1×H100 · BF16", fontsize=11, fontweight="bold", color=INK)
    axL.set_ylim(0, 270)
    for xi, v in [(0, 229.3), (1, 195.9)]:
        axL.text(xi, v + 6, f"{v:.0f}", ha="center", fontsize=10, color=INK)
    axL.annotate("~14% overhead", (1, 150), ha="center", color=GREEN, fontsize=10,
                 fontweight="bold")

    axR.bar([0, 1], [0, 17.3], color=[RED, INDIGO], width=0.6)
    axR.set_xticks([0, 1])
    axR.set_xticklabels(["vanilla", "+ Sluice\n(slots=16)"], fontsize=10)
    axR.set_title("V4-Pro · 4×H100 · FP8 · EP=4", fontsize=11, fontweight="bold", color=INK)
    axR.set_ylim(0, 24)
    axR.text(1, 17.3 + 0.6, "17.3", ha="center", fontsize=10, color=INK)
    axR.annotate("✗ OOM\n(can't run)", (0, 6), ha="center", color=RED, fontsize=10,
                 fontweight="bold")

    for ax in (axL, axR):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Decode throughput — 8 sequences × 128 tokens, greedy, enforce_eager",
                 fontsize=12.5, fontweight="bold", color=INK)
    fig.text(0.5, -0.02, "measured on H100 · with a cache that covers the per-step "
             "working set, streaming costs ~14% · eager mode (no CUDA graphs)",
             ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def slots_sweep_chart(csv_path, path):
    """Plot the measured SLUICE_SLOTS sweep produced by examples/bench_slots.py.

    Reads the CSV (one row per cell) and draws decode tok/s vs slot count, one
    line per batch size, with each batch's resident baseline as a dashed line.
    Points where the cache overflowed or the output diverged from baseline are
    ringed in red — that is the region where the slot count is below the
    per-step working set, so it is not a valid operating point.
    """
    import csv
    from collections import defaultdict

    sweep = defaultdict(list)   # batch -> [(slots, tok_s, overflow, correct)]
    base = {}                   # batch -> resident tok_s
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["status"] != "ok":
                continue
            batch = int(row["batch"])
            tok_s = float(row["decode_tok_s"])
            if row["config"] == "resident":
                base[batch] = tok_s
            else:
                sweep[batch].append((
                    int(row["slots"]), tok_s,
                    row["overflow"] == "1", row["correct"] == "1",
                ))

    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    palette = [INDIGO, TEAL, AMBER, "#B0152F"]
    for i, batch in enumerate(sorted(sweep)):
        color = palette[i % len(palette)]
        pts = sorted(sweep[batch])
        xs = [s for s, *_ in pts]
        ys = [t for _, t, *_ in pts]
        ax.plot(xs, ys, "-o", color=color, lw=2.0, label=f"batch={batch}")
        # Ring invalid points: output diverged from the resident baseline (the
        # slot cache was below the step's working set and dropped experts). This
        # is the reliable per-cell signal; the overflow flag also trips on the
        # discarded startup profiling run, so it is not used here.
        bad = [(s, t) for s, t, ov, ok in pts if not ok]
        if bad:
            ax.scatter([s for s, _ in bad], [t for _, t in bad], s=130,
                       facecolors="none", edgecolors=RED, linewidths=1.8, zorder=5)
        if batch in base:
            ax.axhline(base[batch], ls="--", lw=1.3, color=color, alpha=0.7)
            ax.text(xs[-1], base[batch], f" resident (batch={batch})",
                    va="bottom", ha="right", color=color, fontsize=8)

    ax.scatter([], [], s=130, facecolors="none", edgecolors=RED, linewidths=1.8,
               label="output != baseline (invalid)")
    ax.set_xlabel("SLUICE_SLOTS (resident experts per layer)", fontsize=10)
    ax.set_ylabel("decode throughput (tok/s)", fontsize=10)
    ax.set_title("Decode throughput vs slot count (V2-Lite, 1×H100)",
                 fontsize=12.5, fontweight="bold", color=INK)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "below a step's distinct-expert working set the cache "
             "overflows; above it, throughput flattens to a fixed hook overhead "
             "vs resident", ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def throughput_retained_chart(csv_path, path):
    """Decode throughput as % of the resident baseline, one line per batch,
    VALID (bit-identical) points only. Reads the batch sweep CSV from
    examples/bench_slots.py --short-prompts (decode-isolated). Shows that the
    throughput cost is *streaming*: at full residency every batch lands ~85%
    (routing-hook floor), but below it bigger batches stream more and lose more.
    """
    import csv
    from collections import defaultdict

    series = defaultdict(list)   # batch -> [(slots, pct)]
    for row in csv.DictReader(open(csv_path)):
        if row["status"] != "ok" or row["config"] != "slots" or row["correct"] != "1":
            continue
        series[int(row["batch"])].append(
            (int(row["slots"]), float(row["pct_of_baseline"])))

    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=200)
    ax.axhline(100, ls="--", lw=1.6, color=INK)
    ax.text(64, 101.5, "resident baseline (100%)", ha="right", va="bottom",
            color=INK, fontsize=9)

    palette = [INDIGO, TEAL, AMBER, RED]
    for i, batch in enumerate(sorted(series)):
        color = palette[i % len(palette)]
        pts = sorted(series[batch])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "-o", color=color, lw=2.3, mfc="white", mec=color,
                label=f"batch={batch}")

    ax.annotate("full residency:\nevery batch ~85%\n(routing hook only)",
                (64, 85), (50, 60), fontsize=8.5, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("streaming: bigger batch →\nlarger working set →\nstreams more → loses more",
                (32, 31), (33.5, 12), fontsize=8.5, color="#7A828F")

    ax.set_xlabel("SLUICE_SLOTS (resident experts per layer)", fontsize=10)
    ax.set_ylabel("decode throughput (% of resident)", fontsize=10)
    ax.set_xlim(12, 68)
    ax.set_ylim(0, 112)
    ax.set_title("Throughput cost is streaming, not residency\n"
                 "V2-Lite · 64 experts · short prompts (decode-isolated)",
                 fontsize=12.5, fontweight="bold", color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left",
              title="working set grows with batch")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "valid (bit-identical) points only · gap below 100% = "
             "throughput lost to streaming + the routing hook",
             ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def working_set_chart(path):
    """Measured (V2-Lite, 64 experts, top-6, 1xH100): the distinct experts the
    router selects in one forward step, by step token count (SLUICE_DIAG
    high-water). Shows why SLUICE_SLOTS is gated by prefill, not top_k."""
    _style()
    batches = [1, 2, 4, 8]
    ws = [6, 12, 24, 40]          # measured decode distinct experts per step
    x = list(range(len(batches)))

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(64, ls="--", lw=1.6, color=INK)
    ax.text(len(batches) - 0.5, 64.8, "all 64 experts (full residency)",
            ha="right", va="bottom", color=INK, fontsize=9)
    ax.axhspan(53, 61, color=AMBER, alpha=0.16)
    ax.text(-0.35, 57, "prefill chunk: 53–61\n(the real slot floor)",
            va="center", color="#8A6D1A", fontsize=9, fontweight="bold")

    ax.bar(x, ws, width=0.58, color=TEAL, zorder=3)
    for xi, v in zip(x, ws):
        ax.text(xi, v + 1.6, str(v), ha="center", color=INK, fontweight="bold",
                fontsize=13)
    ax.axhline(6, ls=":", lw=1.3, color=MUTED)
    ax.text(len(batches) - 0.5, 7.6, "top-6 (one token)", ha="right", color=MUTED,
            fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"batch={b}" for b in batches], fontsize=10)
    ax.set_ylabel("distinct experts per decode step (of 64)", fontsize=10)
    ax.set_xlim(-0.6, len(batches) - 0.4)
    ax.set_ylim(0, 70)
    ax.set_title("Decode working set grows ~linearly with batch\n"
                 "V2-Lite · 64 experts · top-6 per token, unioned over the batch",
                 fontsize=12, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "SLUICE_SLOTS must cover a step's union — decode scales "
             "with batch; prefill (a whole chunk) sets the ceiling",
             ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def perf_matrix_chart(csv_path, path):
    """The performance matrix: decode throughput retained (% of resident) as a
    slots x batch heatmap. Valid (bit-identical) cells are coloured by
    throughput; cells where the cache fell below the step's working set and
    dropped experts are greyed (output wrong — fast but invalid)."""
    import csv
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cell, resident = {}, {}
    for row in csv.DictReader(open(csv_path)):
        if row["status"] != "ok":
            continue
        b, tok = int(row["batch"]), float(row["decode_tok_s"])
        if row["config"] == "resident":
            resident[b] = tok
        else:
            cell[(b, int(row["slots"]))] = (
                tok, float(row["pct_of_baseline"]), row["correct"] == "1")
    batches = sorted(resident)
    slots = sorted({s for _, s in cell})
    cols = [("resident", None)] + [("slots", s) for s in slots]

    _style()
    cmap, norm = _retain_cmap(), Normalize(0, 100)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.grid(False)

    for r, b in enumerate(batches):
        for ci, (kind, s) in enumerate(cols):
            tok, pct, ok = (resident[b], 100.0, True) if kind == "resident" \
                else cell[(b, s)]
            if ok:
                tc = "white" if pct >= 52 else INK
                ax.add_patch(Rectangle((ci, r), 0.92, 0.92, facecolor=cmap(norm(pct)),
                             edgecolor="white", lw=3))
                ax.text(ci + 0.46, r + 0.38, f"{tok:.0f}", ha="center", va="center",
                        color=tc, fontsize=16, fontweight="bold")
                ax.text(ci + 0.46, r + 0.66, f"{pct:.0f}%", ha="center",
                        va="center", color=tc, fontsize=9)
            else:
                ax.add_patch(Rectangle((ci, r), 0.92, 0.92, facecolor="#EFF2F5",
                             edgecolor="white", lw=3))
                ax.text(ci + 0.46, r + 0.46, "drops\nexperts", ha="center",
                        va="center", color="#A6AEB9", fontsize=10.5,
                        fontweight="bold", linespacing=1.35)

    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(batches))
    ax.invert_yaxis()
    ax.set_xticks([ci + 0.46 for ci in range(len(cols))])
    ax.set_xticklabels(["resident\n(all 64)"] + [str(s) for s in slots])
    ax.set_yticks([r + 0.46 for r in range(len(batches))])
    ax.set_yticklabels([f"batch {b}" for b in batches])
    ax.set_xlabel("SLUICE_SLOTS  (resident experts per layer)")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Decode throughput (tok/s) retained — slots × batch", loc="left")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("% of resident baseline", fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.text(0.01, -0.02, "V2-Lite · 64 experts · short prompts (decode-isolated).  "
             "grey = slots below the step's working set: experts skipped, output wrong.",
             ha="left", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def serving_frontier_chart(csv_path, path):
    """V4-Pro serving frontier (8xH100, EP=8, measured): output throughput and
    per-token latency (TPOT) vs concurrency. Throughput climbs and saturates
    (knee ~16-32); TPOT degrades as the decode batch grows and streams more
    experts per step. Reads results/v4pro_8xh100_serving_ep8.csv."""
    import csv as _csv
    C, thr, tpot = [], [], []
    for row in _csv.DictReader(open(csv_path)):
        C.append(int(row["concurrency"]))
        thr.append(float(row["output_tok_s"]))
        tpot.append(float(row["mean_tpot_ms"]))
    _style()
    fig, axL = plt.subplots(figsize=(7.8, 4.7))
    axR = axL.twinx()
    axR.grid(False)
    l1 = axL.plot(C, thr, "-o", color=INDIGO, lw=2.4, mfc="white", mec=INDIGO,
                  label="output throughput (tok/s)")
    l2 = axR.plot(C, tpot, "-s", color=AMBER, lw=2.4, mfc="white", mec=AMBER,
                  label="per-token latency TPOT (ms)")
    axL.set_xscale("log", base=2)
    axL.set_xticks(C)
    axL.set_xticklabels([str(c) for c in C])
    axL.set_xlabel("concurrency (in-flight requests)")
    axL.set_ylabel("output token throughput (tok/s)", color=INDIGO)
    axR.set_ylabel("mean TPOT (ms)", color=AMBER)
    axL.tick_params(axis="y", colors=INDIGO)
    axR.tick_params(axis="y", colors=AMBER)
    axL.set_ylim(0, max(thr) * 1.28)
    axR.set_ylim(0, max(tpot) * 1.3)
    for x, y in zip(C, thr):
        axL.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 9),
                     ha="center", color=INDIGO, fontsize=9, fontweight="bold")
    axL.set_title("V4-Pro serving frontier — 8×H100, EP=8 (in 1024 / out 128)",
                  fontsize=12.5, fontweight="bold", color=INK)
    lines = l1 + l2
    axL.legend(lines, [ln.get_label() for ln in lines], fontsize=9, frameon=False,
               loc="upper left")
    axL.spines["top"].set_visible(False)
    axR.spines["top"].set_visible(False)
    fig.text(0.5, -0.02, "throughput climbs with concurrency (knee ~16–32); TPOT "
             "grows as the decode batch streams more experts/step",
             ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def ep_compare_chart(csv_path, path):
    """EP=4 vs EP=8 decode throughput (V4-Pro, measured). EP=8 shards experts
    over 8 ranks: smaller per-rank working set + 2x compute/PCIe -> higher tok/s.
    Reads results/v4pro_ep_compare.csv (cols: ep,batch,decode_tok_s)."""
    import csv as _csv
    from collections import defaultdict
    data = defaultdict(dict)
    for row in _csv.DictReader(open(csv_path)):
        data[int(row["ep"])][int(row["batch"])] = float(row["decode_tok_s"])
    batches = sorted({b for ep in data for b in data[ep]})
    _style()
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    x = list(range(len(batches)))
    w = 0.38
    ep4 = [data[4].get(b, 0) for b in batches]
    ep8 = [data[8].get(b, 0) for b in batches]
    ax.bar([i - w / 2 for i in x], ep4, w, color=GREY, label="EP=4 (4 GPUs)")
    ax.bar([i + w / 2 for i in x], ep8, w, color=INDIGO, label="EP=8 (8 GPUs)")
    for i, (a, b) in enumerate(zip(ep4, ep8)):
        ax.text(i - w / 2, a + 0.5, f"{a:.1f}", ha="center", fontsize=9, color=INK)
        ax.text(i + w / 2, b + 0.5, f"{b:.1f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([f"batch {b}" for b in batches])
    ax.set_ylabel("decode throughput (tok/s)")
    ax.set_ylim(0, max(ep4 + ep8) * 1.2)
    ax.set_title("EP=4 vs EP=8 — V4-Pro decode (slots=16, eager)",
                 fontsize=12.5, fontweight="bold", color=INK)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "EP=8 shards experts over 8 ranks: smaller per-rank "
             "working set + 2× compute/PCIe → higher throughput",
             ha="center", fontsize=7.5, color="#7A828F")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def prefill_chart(csv_path, path):
    """Prefill correct⇄fast tradeoff for V4-Pro (8×H100, EP=8, slots=28): for a
    fixed 4096-token prompt, TTFT and the per-chunk working set vs chunk size
    (max-num-batched-tokens). Small chunks keep each step's set ≤ slots → lossless,
    but a long prompt becomes many chunks → high TTFT. Big chunks → low TTFT but
    the set exceeds the slots → drops experts (lossy). No fast-and-correct: here
    correctness costs ~5×. CSV cols: chunk,ttft_s,ws_per_rank,slots,lossless."""
    import csv as _csv
    rows = sorted((dict(r) for r in _csv.DictReader(open(csv_path))),
                  key=lambda r: int(r["chunk"]))
    chunk = [int(r["chunk"]) for r in rows]
    ttft = [float(r["ttft_s"]) for r in rows]
    wsr = [int(r["ws_per_rank"]) for r in rows]
    good = [r["lossless"] == "1" for r in rows]
    slots = int(rows[0]["slots"])
    LOCAL = 48

    _style()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.7))

    # left: TTFT vs chunk — lossless (green) vs lossy (red)
    axL.plot(chunk, ttft, "-", color=GREY, lw=2.0, zorder=2)
    for c, t, ok in zip(chunk, ttft, good):
        axL.scatter([c], [t], s=140, color=(GREEN if ok else RED), zorder=4,
                    edgecolor="white", linewidth=1.3)
    axL.set_xscale("log", base=2)
    axL.set_xticks(chunk)
    axL.set_xticklabels([str(c) for c in chunk])
    axL.set_ylim(0, max(ttft) * 1.25)
    axL.set_xlabel("prefill chunk size  (max-num-batched-tokens)")
    axL.set_ylabel("TTFT for a 4096-token prompt (s)")
    axL.set_title(f"Correct prefill is slow (slots={slots})", fontsize=12,
                  fontweight="bold", color=INK)
    bi = max((i for i, ok in enumerate(good) if ok), default=None)
    if bi is not None:
        axL.annotate(f"best lossless:\nchunk {chunk[bi]} → {ttft[bi]:.1f} s",
                     (chunk[bi], ttft[bi]), (chunk[bi] * 1.25, ttft[bi] + 0.65),
                     fontsize=8.5, color=GREEN, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=GREEN))
    axL.annotate(f"fastest, but lossy:\nchunk {chunk[-1]} → {ttft[-1]:.2f} s",
                 (chunk[-1], ttft[-1]), (chunk[2], ttft[0] * 0.55), fontsize=8.5,
                 color=RED, ha="right", arrowprops=dict(arrowstyle="->", color=RED))
    axL.scatter([], [], s=110, color=GREEN, label="lossless (set ≤ slots)")
    axL.scatter([], [], s=110, color=RED, label="lossy (drops experts)")
    axL.legend(fontsize=8.5, frameon=False, loc="upper right")
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    # right: per-rank working set vs chunk — crosses the slot budget
    axR.axhspan(0, slots, color=GREEN, alpha=0.07, zorder=0)
    axR.axhspan(slots, LOCAL + 6, color=RED, alpha=0.06, zorder=0)
    axR.plot(chunk, wsr, "-", color=GREY, lw=2.0, zorder=2)
    for c, w, ok in zip(chunk, wsr, good):
        axR.scatter([c], [w], s=140, color=(GREEN if ok else RED), zorder=4,
                    edgecolor="white", linewidth=1.3)
        axR.text(c, w + 1.6, str(w), ha="center", color=INK, fontsize=9, fontweight="bold")
    axR.axhline(slots, ls="--", lw=1.6, color=INK)
    axR.text(chunk[0], slots + 0.9, f"slots that fit VRAM ({slots})", va="bottom",
             ha="left", color=INK, fontsize=8.5, fontweight="bold")
    axR.axhline(LOCAL, ls=":", lw=1.4, color=MUTED)
    axR.text(chunk[-1], LOCAL - 2.6, f"all {LOCAL} experts", va="top", ha="right",
             color=MUTED, fontsize=8.5)
    axR.set_xscale("log", base=2)
    axR.set_xticks(chunk)
    axR.set_xticklabels([str(c) for c in chunk])
    axR.set_ylim(0, LOCAL + 6)
    axR.set_xlabel("prefill chunk size  (max-num-batched-tokens)")
    axR.set_ylabel("distinct experts / rank per chunk")
    axR.set_title("Why: the working set crosses the slot budget", fontsize=12,
                  fontweight="bold", color=INK)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)

    fig.suptitle("Long-context prefill: correct (small chunks) ⇄ fast (big chunks), "
                 "not both — V4-Pro · 8×H100 · EP=8", fontsize=12.5,
                 fontweight="bold", color=INK)
    fig.text(0.5, -0.02, "4096-token prompt. Small chunks keep each step ≤ slots (lossless) but "
             "multiply the step count → high TTFT; big chunks are fast but exceed the slots → drop "
             "experts. Correctness here costs ~5× the TTFT (2.6 s vs 0.48 s).",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def v4_matrix_chart(csv_path, path):
    """V4-Pro decode throughput as a slots × batch matrix (8×H100, EP=8), bracketed
    by OOM on both ends: the resident column OOMs (805 GiB > 640 — all experts can't
    load), and high slot counts OOM (the GPU slot cache itself outgrows VRAM). In
    between, valid cells (slots ≥ the measured per-step per-rank working set) are
    coloured by throughput; cells below the working set drop experts (grey).
    CSV cols: slots,batch,decode_tok_s,working_set,oom."""
    import csv as _csv
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    cell, oom_slots = {}, set()
    for r in _csv.DictReader(open(csv_path)):
        b, s = int(r["batch"]), int(r["slots"])
        if r.get("oom", "0") == "1":
            oom_slots.add(s)
        ws = int(r["working_set"]) if r.get("working_set") else 0
        tok = float(r["decode_tok_s"]) if r.get("decode_tok_s") else 0.0
        cell[(b, s)] = (tok, (s >= ws) if ws else True)
    batches = sorted({b for b, _ in cell})
    slots = sorted({s for _, s in cell})
    cols = [("resident", None)] + [("slots", s) for s in slots]
    many = len(cols) > 7

    _style()
    valid_tok = [t for (_, s), (t, v) in cell.items() if v and s not in oom_slots]
    vmax = max(valid_tok) if valid_tok else 1.0
    cmap, norm = _retain_cmap(), Normalize(0, vmax)
    fig, ax = plt.subplots(figsize=(max(8.6, 0.92 * len(cols) + 1.4), 4.6))
    ax.grid(False)

    def oom_cell(ci, ri, label):
        ax.add_patch(Rectangle((ci, ri), 0.92, 0.92, facecolor="#F3D7DC",
                     edgecolor="white", lw=3))
        ax.text(ci + 0.46, ri + 0.46, label, ha="center", va="center", color=RED,
                fontsize=8 if many else 10.5, fontweight="bold", linespacing=1.25)

    for ri, b in enumerate(batches):
        for ci, (kind, s) in enumerate(cols):
            if kind == "resident":
                oom_cell(ci, ri, "OOM\ncan't\nload" if many else "OOM\ncan't load")
            elif s in oom_slots:
                oom_cell(ci, ri, "OOM\ncache\ntoo big" if many else "OOM\ncache too big")
            else:
                tok, valid = cell[(b, s)]
                if valid:
                    tc = "white" if norm(tok) >= 0.5 else INK
                    ax.add_patch(Rectangle((ci, ri), 0.92, 0.92,
                                 facecolor=cmap(norm(tok)), edgecolor="white", lw=3))
                    ax.text(ci + 0.46, ri + 0.46, f"{tok:.0f}", ha="center",
                            va="center", color=tc, fontsize=12 if many else 16,
                            fontweight="bold")
                else:
                    ax.add_patch(Rectangle((ci, ri), 0.92, 0.92, facecolor="#EFF2F5",
                                 edgecolor="white", lw=3))
                    ax.text(ci + 0.46, ri + 0.46, "drops\nexperts", ha="center",
                            va="center", color="#A6AEB9",
                            fontsize=7.5 if many else 10.5, fontweight="bold",
                            linespacing=1.3)

    ax.set_xlim(0, len(cols))
    ax.set_ylim(0, len(batches))
    ax.invert_yaxis()
    ax.set_xticks([ci + 0.46 for ci in range(len(cols))])
    ax.set_xticklabels(["resident"] + [str(s) for s in slots])
    ax.set_yticks([ri + 0.46 for ri in range(len(batches))])
    ax.set_yticklabels([f"batch {b}" for b in batches])
    ax.set_xlabel("SLUICE_SLOTS (resident experts per layer/rank)  ·  resident = all experts")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("V4-Pro decode throughput (tok/s) — 8×H100, EP=8", loc="left")
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("decode tok/s", fontsize=9)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0)
    fig.text(0.01, -0.03, "Bracketed by OOM both ways: resident can't load 805 GiB "
             "on 640 GiB; high slots OOM as the cache outgrows VRAM. Grey = slots "
             "below the measured per-step working set (experts skipped).",
             ha="left", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    import argparse
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Generate Sluice result charts.")
    parser.add_argument("--slots-csv", default=None,
                        help="CSV from examples/bench_slots.py; renders the "
                             "measured slot-sweep chart in addition to the rest")
    parser.add_argument("--batch-csv", default=None,
                        help="batch-sweep CSV (--short-prompts); renders the "
                             "throughput-retained-per-batch chart")
    parser.add_argument("--serving-csv", default=None,
                        help="V4-Pro serving sweep CSV; renders the serving "
                             "frontier (throughput + TPOT vs concurrency)")
    parser.add_argument("--ep-csv", default=None,
                        help="EP=4-vs-EP=8 CSV (ep,batch,decode_tok_s); renders "
                             "the expert-parallel comparison chart")
    parser.add_argument("--v4-matrix-csv", default=None,
                        help="V4-Pro slots×batch CSV (slots,batch,decode_tok_s,"
                             "working_set); renders the matrix with resident=OOM")
    parser.add_argument("--prefill-csv", default=None,
                        help="V4-Pro prefill CSV (length,ttft_s,ws_per_rank,slots); "
                             "renders the prefill TTFT + working-set chart")
    cli = parser.parse_args()

    # Measured charts (CSV-driven, from real runs) live in their own subdirectory,
    # separate from the estimated/illustrative figures kept in assets/ root.
    measured = os.path.join(here, "measured")
    os.makedirs(measured, exist_ok=True)

    _style()  # one cohesive look across every figure
    comparison_chart(os.path.join(here, "chart-comparison.png"))
    residency_chart(os.path.join(here, "chart-residency.png"))
    tradeoff_chart(os.path.join(here, "chart-tradeoff.png"))
    throughput_chart(os.path.join(here, "chart-throughput.png"))
    working_set_chart(os.path.join(here, "chart-working-set.png"))
    if cli.slots_csv:
        slots_sweep_chart(cli.slots_csv,
                          os.path.join(measured, "chart-slots-sweep.png"))
    if cli.batch_csv:
        perf_matrix_chart(cli.batch_csv,
                          os.path.join(measured, "chart-perf-matrix.png"))
        throughput_retained_chart(cli.batch_csv,
                                  os.path.join(measured, "chart-throughput-retained.png"))
    if cli.serving_csv:
        serving_frontier_chart(cli.serving_csv,
                               os.path.join(measured, "chart-serving-frontier.png"))
    if cli.ep_csv:
        ep_compare_chart(cli.ep_csv,
                         os.path.join(measured, "chart-ep-compare.png"))
    if cli.v4_matrix_csv:
        v4_matrix_chart(cli.v4_matrix_csv,
                        os.path.join(measured, "chart-v4-matrix.png"))
    if cli.prefill_csv:
        prefill_chart(cli.prefill_csv, os.path.join(measured, "chart-prefill.png"))
