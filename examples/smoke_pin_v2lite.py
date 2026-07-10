# SPDX-License-Identifier: Apache-2.0
"""Pin correctness smoke: a static pin must not change model output.

Same greedy decode as bitcompare_v2lite.py, on DeepSeek-V2-Lite with Sluice
streaming active (slots < experts so eviction runs). Token ids go to the file
named by SMOKE_OUT (stdout carries engine logs). Run twice and diff:

    SLUICE_SLOTS=30 SMOKE_OUT=nopin.txt python smoke_pin_v2lite.py
    SLUICE_SLOTS=30 SMOKE_OUT=pin.txt SLUICE_PIN_FILE=pins.json \
        python smoke_pin_v2lite.py
    diff nopin.txt pin.txt && echo PIN_BIT_IDENTICAL

The pin changes expert *placement* only — routing and math are untouched —
so any output difference is a bug in the pin bookkeeping.

Env knobs:
    SMOKE_OUT    (required) output file for the per-prompt token ids
    MODEL        model id (default deepseek-ai/DeepSeek-V2-Lite)
    EAGER        0 lifts enforce_eager (pair with SLUICE_GRAPH=1 or PIECEWISE)
    PIECEWISE    1 sets cudagraph_mode=PIECEWISE (pair with SLUICE_PIECEWISE=1)
    MAX_BT       max_num_batched_tokens (default 1024; keep small — see below)
    MAX_SEQS     max_num_seqs (default 8)
    GPU_MEM      gpu_memory_utilization (default 0.5)
    MOE_BACKEND  MoE kernel backend (default triton; must apply expert_map)
    SLUICE_*     the arm under test (SLUICE_SLOTS, SLUICE_PIN_FILE, ...)"""

import os

from vllm import LLM, SamplingParams

PROMPTS = [
    "The capital of France is",
    "Q: What is 2+2? A:",
    "Once upon a time",
    "def fibonacci(n):",
    "The three laws of robotics state",
    "Water boils at a temperature of",
    "In 1969, humanity first",
    "Translate to French: good morning",
]


def main() -> None:
    llm = LLM(
        model=os.environ.get("MODEL", "deepseek-ai/DeepSeek-V2-Lite"),
        trust_remote_code=True,
        enforce_eager=os.environ.get("EAGER", "1") != "0",
        max_model_len=1024,
        # Small prefill chunks + one sequence at a time keep every step's
        # working set under the slot count, so no step runs in waves and the
        # pin/no-pin bit-compare is meaningful (wave partitioning differs
        # between configs by design and is exact-but-not-bit-identical).
        max_num_batched_tokens=int(os.environ.get("MAX_BT", "1024")),
        max_num_seqs=int(os.environ.get("MAX_SEQS", "8")),
        gpu_memory_utilization=float(os.environ.get("GPU_MEM", "0.5")),
        moe_backend=os.environ.get("MOE_BACKEND", "triton"),
        **(
            {"compilation_config": {"cudagraph_mode": "PIECEWISE"}}
            if os.environ.get("PIECEWISE") == "1"
            else {}
        ),
    )
    sp = SamplingParams(temperature=0.0, max_tokens=48, ignore_eos=True)
    outs = llm.generate(PROMPTS, sp)
    # Results go to a dedicated file: engine INFO lines share stdout/stderr,
    # and the bit-compare diff must see token ids only.
    with open(os.environ["SMOKE_OUT"], "w") as f:
        for o in outs:
            f.write(f"{o.prompt!r} {list(o.outputs[0].token_ids)}\n")
    print("SMOKE_RUN_DONE", flush=True)


if __name__ == "__main__":
    main()
