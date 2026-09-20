#!/usr/bin/env python3
"""JEV-style structured reads over the DiffusionGemma visual server.

Unstructured state in, typed probabilistic decisions out: a system prompt lists
questions with single-token choices, the answer template has one '@' slot per
question, and the server returns the argmax label and its entropy per slot.

Usage:
  python3 jev_reads_example.py --model diffusiongemma-26B-A4B-it-Q4_K_M.gguf \
      --ngl 99 --cpu-moe 30 --maxtok 2048 [--task triage|language|unit]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile


class JevClient:
    def __init__(self, argv: list[str]) -> None:
        self.proc = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1
        )
        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"server did not start: {ready!r}")
        _, n_vocab, maxtok = ready.split()
        self.n_vocab = int(n_vocab)
        self.maxtok = int(maxtok)

    def read(self, system: str, state: str, template: str, seed: int = 0, max_steps: int = 1,
             labels: list[str] | None = None) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        req = {
            "seed": seed,
            "read_only": True,
            "max_steps": max_steps,
            "canvas_template": template,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": state},
            ],
        }
        if labels:
            req["labels"] = list(labels)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(req, f)
            path = f.name
        try:
            self.proc.stdin.write(path + "\n")
            self.proc.stdin.flush()
            result = None
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("server closed")
                if line.startswith("R "):
                    result = json.loads(line[2:])
                elif line.startswith("ERR"):
                    raise RuntimeError(line.strip())
                elif line.startswith("DONE"):
                    break
            if result is None:
                raise RuntimeError("no read returned")
            return result
        finally:
            os.unlink(path)

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
        self.proc.wait(timeout=30)


def run_task(client: JevClient, system: str, state: str, template: str, questions: list[tuple[str, list[str]]], max_steps: int) -> None:
    labels: list[str] = []
    for _, options in questions:
        for option in options:
            if option not in labels:
                labels.append(option)
    out = client.read(system, state, template, max_steps=max_steps, labels=labels)
    slots = out["slots"][: len(questions)]  # the canvas is padded; only the template's slots matter
    if len(slots) != len(questions):
        print(f"  warning: {len(slots)} slots for {len(questions)} questions")
    print(f"  text: {out['text'].strip()[:110]!r}")
    for (name, options), slot in zip(questions, slots):
        lp = slot.get("logprobs")
        if not lp or len(lp) != len(labels):
            print(f"  {name:10s} = {slot['text'].strip()!r:14s} entropy={slot['entropy']:.3f} (no logprobs)")
            continue
        vals = [lp[labels.index(o)] for o in options]
        m = max(vals)
        ex = [math.exp(v - m) for v in vals]
        s = sum(ex)
        probs = [e / s for e in ex]
        best = max(range(len(options)), key=lambda k: probs[k])
        print(f"  {name:10s} = {options[best]!r:14s} p={probs[best]:.3f} entropy={slot['entropy']:.3f}")


def sys_prompt(questions: list[tuple[str, list[str]]]) -> str:
    lines = ["Answer each question with exactly one label from the allowed values.", ""]
    for name, options in questions:
        lines.append(f"Question {name}: {' / '.join(options)}")
    lines.append("")
    lines.append("Reply with one line per question, formatted as \"name: label\".")
    return "\n".join(lines)


def task_triage() -> tuple:
    questions = [
        ("urgent", ["yes", "no"]),
        ("bucket", ["outage", "billing", "feature", "other"]),
        ("tone", ["calm", "annoyed", "furious"]),
    ]
    template = "<|channel>thought\n<channel|>urgent: @\nbucket: @\ntone: @"
    state = (
        "Ticket: Since this morning the dashboard shows a blank page after login. "
        "This is blocking our entire support team and we have no workaround. "
        "We are extremely frustrated, this is the third outage this week."
    )
    return questions, template, state


def task_language() -> tuple:
    questions = [("language", ["python", "rust", "java", "go", "sql"])]
    template = "language: @"
    state = (
        "def quicksort(xs):\n"
        "    if len(xs) <= 1: return xs\n"
        "    p = xs[len(xs) // 2]\n"
        "    return quicksort([x for x in xs if x < p]) + [x for x in xs if x == p] + quicksort([x for x in xs if x > p])"
    )
    return questions, template, state


def task_unit() -> tuple:
    questions = [("comparison", ["first", "second", "equal"])]
    template = "comparison: @"
    state = "Compare the two values: 12 meters and 39 feet. Which is larger?"
    return questions, template, state


TASKS = {"triage": task_triage, "language": task_language, "unit": task_unit}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="", help="path to llama-diffusion-gemma-visual-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--cpu-moe", type=int, default=0)
    ap.add_argument("--maxtok", type=int, default=2048)
    ap.add_argument("--canvas", type=int, default=32, help="canvas width for the reads (model default is 256)")
    ap.add_argument("--max-steps", type=int, default=1, help="denoising steps before the read (1 = one forward)")
    ap.add_argument("--task", choices=sorted(TASKS) + ["all"], default="all")
    args = ap.parse_args()

    server = args.server or os.path.join(os.path.dirname(__file__), "..", "..", "build", "bin",
                                         "llama-diffusion-gemma-visual-server")
    os.environ["NGL"] = str(args.ngl)
    os.environ["MAXTOK"] = str(args.maxtok)
    os.environ["DG_CANVAS"] = str(args.canvas)
    if args.cpu_moe > 0:
        os.environ["DG_N_CPU_MOE"] = str(args.cpu_moe)

    names = sorted(TASKS) if args.task == "all" else [args.task]
    client = JevClient([server, args.model])
    try:
        for name in names:
            questions, template, state = TASKS[name]()
            print(f"[{name}]")
            run_task(client, sys_prompt(questions), state, template, questions, args.max_steps)
            print()
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
