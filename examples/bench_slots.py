# SPDX-License-Identifier: Apache-2.0
"""Sweep SLUICE_SLOTS and measure decode throughput — the performance matrix.

Produces a (slots x batch-size) matrix of decode throughput for one MoE model
(default DeepSeek-V2-Lite, which fits a single GPU either way, so the offloaded
runs can be compared head-to-head against the resident baseline). Each cell
records:

  * decode tok/s and its fraction of the resident (no-Sluice) baseline,
  * whether the slot cache *overflowed* — the router selected more distinct
    local experts in a single step than there were slots, so experts were
    skipped (Sluice logs a warning; we surface it as a column), and
  * whether the output token ids still match the baseline bit-for-bit
    (they should, unless the cache overflowed and dropped experts).

Why a subprocess per cell: the Sluice plugin reads ``SLUICE_SLOTS`` once, at
``register()`` time, and monkeypatches vLLM's offloader factory — so the slot
count is fixed for a process's lifetime. Each (slots, batch) cell therefore runs
in a fresh ``--worker`` process; this also hands every run a clean GPU.

Run (on the GPU box):

    python examples/bench_slots.py                      # 6..64 slots, batch 1 & 8
    python examples/bench_slots.py --slots 6,16,32,64 --batch 1,4,8,16
    python examples/bench_slots.py --max-tokens 128 --out results/slots.csv

Unquantized V2-Lite needs an ``expert_map``-honoring backend; on Hopper vLLM
already prefers TRITON. If your build defaults to FlashInfer, force it with
``--moe-backend triton`` (NVFP4/FP8 checkpoints want ``marlin``). Then chart it:

    python assets/make_charts.py --slots-csv results/slots.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict

# Varied prompts so different sequences in a batch route to *different* experts,
# giving a realistic per-step working set (identical prompts would all select
# the same top_k experts and never exercise the cache). Tiled up to batch size.
PROMPTS = [
    "The capital of France is",
    "Q: What is 2+2? A:",
    "Once upon a time",
    "In 1969, humanity first",
    "The mitochondria is the",
    "def fibonacci(n):",
    "Translate to French: good morning",
    "The three laws of robotics state",
    "Water boils at a temperature of",
    "The opening line of Moby Dick is",
    "A haiku about autumn:",
    "The derivative of x squared is",
    "Breaking news: scientists have discovered",
    "The recipe calls for two cups of",
    "import numpy as np\n",
    "To be, or not to be, that is",
]

# Single-token prompts: minimal prefill, so the per-step working set is set by
# decode (the running batch), not by a long prompt's prefill. Distinct words so
# the sequences diverge and exercise diverse routing.
SHORT_PROMPTS = [
    "The", "A", "In", "I", "It", "We", "He", "She",
    "They", "You", "When", "What", "Why", "How", "Now", "Once",
]

# A real (non-warmup) step ran multi-wave: computed exactly, but not
# bit-identical to a single launch (float summation order changes).
OVERFLOW_MARKER = "executing in waves"


@dataclass
class CellResult:
    batch: int
    config: str          # "resident" or "slots"
    slots: int           # 0 for the resident baseline
    decode_tok_s: float
    total_tok_s: float
    ids_hash: str
    overflow: bool
    status: str          # "ok" or "FAIL"


# --------------------------------------------------------------------------- #
# Worker: one vLLM instance, one (slots, batch) measurement.                   #
# --------------------------------------------------------------------------- #

def _make_prompts(batch: int, short: bool = False) -> list[str]:
    pool = SHORT_PROMPTS if short else PROMPTS
    return [pool[i % len(pool)] for i in range(batch)]


def run_worker(args: argparse.Namespace) -> None:
    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem,
    )
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    # One model load (the expensive step with offloading), then every batch size
    # is measured in-process. SLUICE_SLOTS is fixed for the process, so slots is
    # the outer (per-process) axis and batch is swept here.
    llm = LLM(**llm_kwargs)

    greedy = lambda n: SamplingParams(temperature=0.0, max_tokens=n, ignore_eos=True)

    for batch in args.batch:  # a list of ints in worker mode
        prompts = _make_prompts(batch, args.short_prompts)

        # Warm up so the timed run measures steady-state decode, not first-touch
        # CPU->GPU streaming or compilation.
        llm.generate(prompts, greedy(8))

        # Decode isolation: time (prefill + 1 token), then (prefill + N tokens);
        # the difference is N-1 tokens of pure decode, robust to prompt lengths.
        t0 = time.perf_counter()
        llm.generate(prompts, greedy(1))
        t1 = time.perf_counter()
        tA = time.perf_counter()
        out = llm.generate(prompts, greedy(args.max_tokens))
        tB = time.perf_counter()

        decode_time = max((tB - tA) - (t1 - t0), 1e-9)
        decode_tokens = batch * max(args.max_tokens - 1, 0)
        decode_tok_s = decode_tokens / decode_time if decode_tokens else 0.0
        total_tok_s = (batch * args.max_tokens) / max(tB - tA, 1e-9)

        ids = [list(o.outputs[0].token_ids) for o in out]
        ids_hash = hashlib.sha1(repr(ids).encode()).hexdigest()[:12]

        print("BENCH_RESULT " + json.dumps({
            "batch": batch,
            "decode_tok_s": round(decode_tok_s, 2),
            "total_tok_s": round(total_tok_s, 2),
            "ids_hash": ids_hash,
        }), flush=True)


# --------------------------------------------------------------------------- #
# Driver: launch one worker per cell, assemble the matrix.                     #
# --------------------------------------------------------------------------- #

def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.replace(" ", "").split(",") if x]


def _launch(args: argparse.Namespace, slots: int, sluice: bool) -> list[CellResult]:
    """Run one worker process (one model load) that measures every batch size,
    and return one CellResult per batch."""
    env = os.environ.copy()
    if sluice:
        env["SLUICE_SLOTS"] = str(slots)
    else:
        env.pop("SLUICE_SLOTS", None)

    argv = [
        sys.executable, os.path.abspath(__file__), "--worker",
        "--model", args.model,
        "--batch", args.batch,            # full comma list; worker sweeps it
        "--max-tokens", str(args.max_tokens),
        "--max-model-len", str(args.max_model_len),
        "--gpu-mem", str(args.gpu_mem),
    ]
    if args.moe_backend:
        argv += ["--moe-backend", args.moe_backend]
    if args.short_prompts:
        argv += ["--short-prompts"]

    config = "slots" if sluice else "resident"
    label = f"slots={slots}" if sluice else "resident"
    print(f"[{label}] launching (batches {args.batch}) ...", flush=True)
    proc = subprocess.run(argv, env=env, capture_output=True, text=True)

    # Overflow is detected per-process (the offloader warns once per layer); the
    # authoritative per-cell validity signal is `correct` (ids vs baseline).
    overflow = OVERFLOW_MARKER in proc.stderr
    parsed = [
        json.loads(line[len("BENCH_RESULT "):])
        for line in proc.stdout.splitlines()
        if line.startswith("BENCH_RESULT ")
    ]

    if proc.returncode != 0 or not parsed:
        tail = "\n".join(proc.stderr.strip().splitlines()[-8:])
        print(f"   FAILED (rc={proc.returncode}):\n{tail}", flush=True)
        return [CellResult(b, config, slots, 0.0, 0.0, "", overflow, "FAIL")
                for b in _parse_int_list(args.batch)]

    cells = [
        CellResult(d["batch"], config, slots, d["decode_tok_s"],
                   d["total_tok_s"], d["ids_hash"], overflow, "ok")
        for d in parsed
    ]
    for c in cells:
        print(f"   batch={c.batch}: {c.decode_tok_s:.0f} tok/s", flush=True)
    return cells


def run_driver(args: argparse.Namespace) -> None:
    slots_list = _parse_int_list(args.slots)
    batch_list = _parse_int_list(args.batch)

    # The resident baseline itself is not process-deterministic at batch>1
    # (engine batching timing -> reduction shapes -> rare greedy argmax
    # flips), so run it twice and accept a match against either realization.
    print("[resident baseline]", flush=True)
    base_cells = _launch(args, slots=0, sluice=False)
    print("[resident baseline - repeat, greedy stability check]", flush=True)
    base_cells2 = _launch(args, slots=0, sluice=False)
    baseline = {c.batch: c for c in base_cells}
    base_hashes: dict[int, set] = {}
    for c in base_cells + base_cells2:
        if c.status == "ok" and c.ids_hash:
            base_hashes.setdefault(c.batch, set()).add(c.ids_hash)
    for b in sorted(base_hashes):
        if len(base_hashes[b]) > 1:
            print(f"   note: baseline nondeterministic at batch={b}; "
                  "correct = matches either realization", flush=True)
    rows: list[CellResult] = list(base_cells)

    for slots in slots_list:
        rows += _launch(args, slots=slots, sluice=True)

    _write_csv(rows, baseline, base_hashes, args.out)
    _print_matrix(rows, slots_list, batch_list, baseline, base_hashes)


def _pct(cell: CellResult, base: CellResult) -> float:
    if base.status != "ok" or base.decode_tok_s <= 0:
        return 0.0
    return 100.0 * cell.decode_tok_s / base.decode_tok_s


def _correct(cell: CellResult, base: CellResult, base_hashes=None) -> bool:
    if cell.status != "ok" or base is None or base.status != "ok":
        return False
    accepted = (base_hashes or {}).get(cell.batch) or {base.ids_hash}
    return cell.ids_hash in accepted


def _write_csv(rows: list[CellResult], baseline: dict[int, CellResult],
               base_hashes: dict, path: str) -> None:
    out_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(out_dir, exist_ok=True)
    header = ("batch,config,slots,decode_tok_s,total_tok_s,"
              "pct_of_baseline,overflow,correct,status")
    lines = [header]
    for r in rows:
        base = baseline.get(r.batch)
        pct = _pct(r, base) if base else 0.0
        correct = _correct(r, base, base_hashes) if base else True
        lines.append(
            f"{r.batch},{r.config},{r.slots},{r.decode_tok_s},{r.total_tok_s},"
            f"{pct:.1f},{int(r.overflow)},{int(correct)},{r.status}"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {path} ({len(rows)} rows)")


def _print_matrix(rows, slots_list, batch_list, baseline, base_hashes=None) -> None:
    by_key = {(r.batch, r.slots, r.config): r for r in rows}
    col_w = 16
    lbl_w = 10

    print("\nDecode throughput  tok/s (% of resident baseline)")
    print("  ! = ran multi-wave (exact math, not bit-identical)   "
          "x = output != baseline   FAIL = run died\n")
    header = "slots".ljust(lbl_w) + "".join(f"batch={b}".ljust(col_w) for b in batch_list)
    print(header)
    print("-" * len(header))

    def fmt(cell, base):
        if cell is None:
            return "-"
        if cell.status == "FAIL":
            return "FAIL"
        flags = ("!" if cell.overflow else "") + ("" if _correct(cell, base, base_hashes) else "x")
        return f"{cell.decode_tok_s:.0f} ({_pct(cell, base):.0f}%){flags}"

    base_row = "resident".ljust(lbl_w)
    for b in batch_list:
        base_row += fmt(baseline.get(b), baseline.get(b)).ljust(col_w)
    print(base_row)

    for s in slots_list:
        line = str(s).ljust(lbl_w)
        for b in batch_list:
            line += fmt(by_key.get((b, s, "slots")), baseline.get(b)).ljust(col_w)
        print(line)


# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    p.add_argument("--slots", default="6,8,12,16,24,32,48,64",
                   help="comma-separated SLUICE_SLOTS values to sweep")
    p.add_argument("--batch", default="1,8",
                   help="comma-separated decode batch sizes (sequences)")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-mem", type=float, default=0.5)
    p.add_argument("--moe-backend", default=None,
                   help="e.g. triton (unquantized) or marlin (NVFP4/FP8); "
                        "omit to let vLLM choose")
    p.add_argument("--short-prompts", action="store_true",
                   help="single-token prompts (minimal prefill) to isolate the "
                        "decode working set from the prefill working set")
    p.add_argument("--out", default="results/slots_sweep.csv")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    if args.worker:
        args.batch = _parse_int_list(args.batch)  # worker sweeps all batch sizes
        run_worker(args)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
