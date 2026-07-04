# SPDX-License-Identifier: Apache-2.0
"""Standalone: Sluice (H100, offload) vs H200-resident comparison + TCO positioning.

Kept separate from make_charts.py on purpose. Reads:
  results/h200_resident_inferencex.csv   (InferenceX, 8xH200 vLLM FP8 EP=8, 8192/1024)
  results/v4pro_h100_sluice_longctx.csv  (Sluice 8xH100 FP8 EP=8 slots=16, ISL=8192)

    python assets/make_compare_charts.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

TEAL, INDIGO, AMBER, GREY, INK = "#0DB8AB", "#4338CA", "#F5A926", "#C9CED6", "#1F2430"
RED, GREEN, MUTED, HAIR = "#B0152F", "#0B7A33", "#6B7480", "#E7EBEF"


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
        "font.size": 12, "text.color": INK,
        "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlepad": 14,
        "axes.labelsize": 11, "axes.labelcolor": "#2B313B",
        "axes.edgecolor": HAIR, "axes.linewidth": 1.2,
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": HAIR, "grid.linewidth": 1.0,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "figure.dpi": 200, "savefig.dpi": 200,
    })


def _rows(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def compare_chart(h200_csv, sluice_csv, path):
    h = _rows(h200_csv)
    hc = [int(r["concurrency"]) for r in h]
    hi = [float(r["interactivity_tok_s_user"]) for r in h]
    s = [r for r in _rows(sluice_csv) if r.get("note") == "ok"]
    sc = [int(r["batch"]) for r in s]
    si = [float(r["per_user_tok_s"]) for r in s]

    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(hc, hi, "-o", color=INDIGO, lw=2.6, mfc="white", mec=INDIGO,
            label="8×H200 node — resident (1128 GiB > 805, fits)")
    ax.plot(sc, si, "-s", color=TEAL, lw=2.6, mfc="white", mec=TEAL,
            label="8×H100 node + Sluice (640 GiB < 805, offload)")
    # flat guide for Sluice (stays ~4–5 tok/s/user)
    ax.axhline(sum(si) / len(si), xmin=0.02, xmax=0.98, ls=":", lw=1.4, color=TEAL, alpha=0.6)
    ax.axhspan(5, 10, color=GREEN, alpha=0.07)
    ax.text(64, 7.5, "comfortable interactive\n(~5–10 tok/s/user)", ha="right",
            va="center", color=GREEN, fontsize=8.5)
    for x, y in zip(sc, si):
        ax.annotate(f"{y:.1f}", (x, y), textcoords="offset points", xytext=(0, -14),
                    ha="center", color=TEAL, fontsize=9, fontweight="bold")
    for x, y in [(hc[0], hi[0]), (hc[3], hi[3])]:
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", color=INDIGO, fontsize=9, fontweight="bold")
    ax.annotate("≈10× faster per user", (1, 44.5), (2.2, 33), color=INK, fontsize=9.5,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=INK))
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])
    ax.set_xlabel("concurrency  (≈ in-flight requests)")
    ax.set_ylabel("per-user decode speed (tok/s/user)")
    ax.set_ylim(0, 50)
    ax.set_title("V4-Pro per-user speed — 8×H100+Sluice vs 8×H200-resident  (8192-token context)",
                 fontsize=12.5, fontweight="bold", color=INK)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(0.5, -0.03, "Same vLLM / FP8 / EP=8, one 8-GPU node each (TP=8). An 8×H200 node "
             "(8×141 = 1128 GiB) fits V4-Pro (805 GiB) resident; an 8×H100 node (640 GiB) does "
             "not — Sluice streams experts from host RAM. Resident is ~5–10× faster per user; "
             "Sluice's value is running it on the 8×H100 node you already own, not speed. "
             "(Sluice's long-context prefill is lossy at slots=16.)",
             ha="center", fontsize=7.3, color=MUTED, wrap=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


def tco_chart(path):
    """Positioning matrix: the argument that actually says 'it helps'."""
    _style()
    cols = ["8×H100 alone", "8×H100 + Sluice", "8×H200 / GB200 resident"]
    rows = ["GPU memory / node", "Runs V4-Pro (805 GiB)?", "Per-user speed",
            "Aggregate throughput"]
    cells = [
        ["640 GiB", "640 GiB + host RAM", "1128 GiB (8×H200)"],
        ["✗  OOM", "✓  yes", "✓  yes"],
        ["—", "~4–5 tok/s/user", "~22–77 tok/s/user"],
        ["—", "~50 tok/s (1 node)", "~400+ tok/s (1 node)"],
    ]
    colcolor = ["#F3D7DC", "#E8F6F4", "#E9E7FA"]   # red-ish / teal-ish / indigo-ish
    headcolor = [RED, TEAL, INDIGO]
    fig, ax = plt.subplots(figsize=(9.4, 3.9))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(rows) + 1)
    ax.axis("off")
    ax.invert_yaxis()
    # header
    for c in range(3):
        ax.add_patch(Rectangle((c + 1, 0), 0.96, 0.92, facecolor=headcolor[c]))
        ax.text(c + 1.48, 0.46, cols[c], ha="center", va="center", color="white",
                fontsize=10.5, fontweight="bold")
    for r, name in enumerate(rows):
        ax.text(0.95, r + 1.46, name, ha="right", va="center", color=INK,
                fontsize=10.5, fontweight="medium")
        for c in range(3):
            ax.add_patch(Rectangle((c + 1, r + 1), 0.96, 0.92, facecolor=colcolor[c],
                         edgecolor="white", lw=3))
            txt = cells[r][c]
            col = (GREEN if txt.startswith("✓") else RED if txt.startswith("✗")
                   else INK)
            ax.text(c + 1.48, r + 1.46, txt, ha="center", va="center", color=col,
                    fontsize=10, fontweight="bold" if txt[0] in "✓✗" else "normal")
    ax.set_title("Where Sluice helps: serve V4-Pro on the 8×H100 node you already own",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    print("wrote", path)


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    out = os.path.join(here, "measured")
    os.makedirs(out, exist_ok=True)
    compare_chart(os.path.join(root, "results/h200_resident_inferencex.csv"),
                  os.path.join(root, "results/v4pro_h100_sluice_longctx.csv"),
                  os.path.join(out, "chart-compare-h200.png"))
    tco_chart(os.path.join(out, "chart-tco.png"))
