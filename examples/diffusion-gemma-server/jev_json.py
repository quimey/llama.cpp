#!/usr/bin/env python3
"""Interactive JEV reads: typed questions in, probabilistic decisions out.

Spawns the DiffusionGemma visual server once, then reads one JSON request per
line on stdin and writes one JSON response per line on stdout, so it can be
used from a terminal or piped from a script.

Request:
  {"state": "<unstructured state>",
   "questions": [
     {"key": "urgent",   "type": "bool"},
     {"key": "bucket",   "type": "enum", "options": ["outage","billing","feature","other"]},
     {"key": "severity", "type": "scale", "min": 1, "max": 4}
   ],
   "max_steps": 1,            // optional, denoising passes before the read
   "seed": 0,                 // optional
   "system": "...",           // optional, overrides the generated system prompt
   "template": "...",         // optional, overrides the generated canvas template
   "include_text": false      // optional, include the decoded canvas
  }

Response:
  {"decisions": {"urgent": {"label": "yes", "p": 0.91, "options": {...}}, ...},
   "stats": {"latency_ms": ..., "gen_ms": ..., "passes": 1, "canvas_n": 32,
             "prompt_n": 87, "slots": 3, "canvas_tokens_per_s": ..., "model_load_ms": ...}}

Usage:
  python3 jev_json.py --model diffusiongemma-26B-A4B-it-Q4_K_M.gguf --ngl 99 --canvas 32
  echo '{"state":"...","questions":[...]}' | python3 jev_json.py --model ...
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time


def question_options(q: dict) -> list[str]:
    t = q.get("type", "bool")
    if t == "bool":
        return q.get("options", ["yes", "no"])
    if t in ("enum", "choices"):
        return list(q["options"])
    if t == "scale":
        return [str(v) for v in range(int(q["min"]), int(q["max"]) + 1)]
    raise ValueError(f"unknown question type: {t}")


def build_prompt(questions: list[dict]) -> str:
    lines = ["Answer each question with exactly one label from the allowed values.", ""]
    for q in questions:
        lines.append(f"Question {q['key']}: {' / '.join(question_options(q))}")
    lines.append("")
    lines.append('Reply with one line per question, formatted as "key: label".')
    return "\n".join(lines)


class JevServer:
    def __init__(self, argv: list[str]) -> None:
        t0 = time.time()
        self.proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        assert self.proc.stdout is not None
        ready = self.proc.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"server did not start: {ready!r}")
        _, n_vocab, maxtok = ready.split()
        self.n_vocab = int(n_vocab)
        self.maxtok = int(maxtok)
        self.load_ms = (time.time() - t0) * 1000.0

    def read(self, messages: list[dict], template: str, labels: list[str], seed: int, max_steps: int) -> dict:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        req = {
            "seed": seed,
            "read_only": True,
            "max_steps": max_steps,
            "canvas_template": template,
            "labels": labels,
            "messages": messages,
        }
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


def decide(server: JevServer, req: dict) -> dict:
    questions = req["questions"]
    state = req.get("state", "")
    max_steps = int(req.get("max_steps", 1))
    seed = int(req.get("seed", 0))

    system = req.get("system") or build_prompt(questions)
    template = req.get("template") or ("<|channel>thought\n<channel|>" + "\n".join(f"{q['key']}: @" for q in questions))

    labels: list[str] = []
    for q in questions:
        for o in question_options(q):
            if o not in labels:
                labels.append(o)

    t0 = time.time()
    out = server.read(
        [{"role": "system", "content": system}, {"role": "user", "content": state}],
        template, labels, seed, max_steps,
    )
    latency_ms = (time.time() - t0) * 1000.0

    slots = out["slots"]
    decisions: dict[str, dict] = {}
    for q, slot in zip(questions, slots[: len(questions)]):
        options = question_options(q)
        lp = slot.get("logprobs")
        if not lp or len(lp) != len(labels):
            decisions[q["key"]] = {"label": slot["text"].strip(), "p": None, "options": {}, "entropy": slot["entropy"]}
            continue
        vals = [lp[labels.index(o)] for o in options]
        m = max(vals)
        ex = [math.exp(v - m) for v in vals]
        s = sum(ex)
        probs = [e / s for e in ex]
        best = max(range(len(options)), key=lambda k: probs[k])
        decisions[q["key"]] = {
            "label": options[best],
            "p": round(probs[best], 4),
            "options": {o: round(p, 4) for o, p in zip(options, probs)},
            "entropy": round(slot["entropy"], 4),
        }

    st = out.get("stats", {})
    canvas_n = st.get("canvas_n", len(out.get("canvas", [])))
    gen_ms = st.get("gen_ms", 0.0)
    stats = {
        "latency_ms": round(latency_ms, 1),
        "gen_ms": round(gen_ms, 1),
        "model_load_ms": round(server.load_ms, 1),
        "passes": st.get("passes", max_steps),
        "prompt_n": st.get("prompt_n"),
        "canvas_n": canvas_n,
        "slots": len(questions),
        "canvas_tokens_per_s": round(canvas_n / (gen_ms / 1000.0), 1) if gen_ms > 0 else None,
    }
    resp: dict = {"decisions": decisions, "stats": stats}
    if req.get("include_text"):
        resp["text"] = out.get("text")
    return resp


DEMO = {
    "state": "Since this morning the dashboard shows a blank page after login. This is blocking our "
             "entire support team and we have no workaround. We are extremely frustrated, this is the "
             "third outage this week.",
    "questions": [
        {"key": "urgent", "type": "bool"},
        {"key": "bucket", "type": "enum", "options": ["outage", "billing", "feature", "other"]},
        {"key": "severity", "type": "scale", "min": 1, "max": 4},
    ],
    "max_steps": 1,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.path.join(os.path.dirname(__file__), "..", "..", "build", "bin",
                                                     "llama-diffusion-gemma-visual-server"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--ngl", type=int, default=99)
    ap.add_argument("--cpu-moe", type=int, default=0)
    ap.add_argument("--maxtok", type=int, default=2048)
    ap.add_argument("--canvas", type=int, default=32)
    ap.add_argument("--demo", action="store_true", help="run one built-in request and exit")
    args = ap.parse_args()

    os.environ["NGL"] = str(args.ngl)
    os.environ["MAXTOK"] = str(args.maxtok)
    os.environ["DG_CANVAS"] = str(args.canvas)
    if args.cpu_moe > 0:
        os.environ["DG_N_CPU_MOE"] = str(args.cpu_moe)

    server = JevServer([args.server, args.model])
    print(json.dumps({"ready": True, "n_vocab": server.n_vocab, "maxtok": server.maxtok,
                      "canvas": args.canvas, "model_load_ms": round(server.load_ms, 1)}), flush=True)
    try:
        if args.demo:
            print(json.dumps(decide(server, DEMO)), flush=True)
            return 0
        for line in sys.stdin:
            line = line.strip()
            if not line or line in ("quit", "exit"):
                continue
            try:
                print(json.dumps(decide(server, json.loads(line))), flush=True)
            except Exception as e:  # noqa: BLE001
                print(json.dumps({"error": str(e)}), flush=True)
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
