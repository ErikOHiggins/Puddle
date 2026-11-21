"""Does a change to strata-poly leave a model's answer as it was?  Two engines, one prompt, greedy: the token ids
each writes must be the same, and each engine must give the same ids every time it runs.

For each run config (strata-<model>.json) the prompt is tokenized with the config's tokenizer (as text: no chat
template), and each engine runs it in its one-shot mode with the config's arguments, as the server starts it
(serve/server.py engine_args() and child_env(): the cards, the layer split, the libraries), plus
--expert-swaps 0 (the expert cache stays as it was loaded, so a run does not depend on the timing of the copies),
--temperature 0, --tokens and --max-new.  --expert-profile is left out, so no run writes the model's profile.
The engines run with STRATA_FREEZE_EXPERTS=1 (--no-freeze leaves it out): with experts on the CPU, the split between
the card's experts and the CPU's, and the way a window of guesses goes through the layers (whole or in halves, by the
times measured so far), decide a layer's sums to the last bit, and the freeze holds both still for the run.  On
echo Laguna-S's two runs of one engine parted from token 16 without it.

    .venv/bin/python tools/engine_regress.py --old /tmp/strata-poly-old strata-laguna-xs21-q4_k_m.json ...

--new is the config's own engine unless given.  The server must not be running (the runs need the cards' VRAM).
Exit 0: every model's ids agree; 1: a difference, printed with the first place it starts; 2: a run failed.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT = ("The history of the printing press begins in the fifteenth century, when Johannes Gutenberg combined a "
          "screw press with movable metal type. Explain, step by step, why this mattered for science, then write a "
          "short Python function that counts the words in a sentence.")
DROP = {"--expert-profile": 1, "--expert-profile-save": 1}   # flag: how many values follow it


def regress_args(args: list[str], ids: list[int], max_new: int) -> list[str]:
    """The config's engine arguments for one deterministic one-shot run."""
    out, i = [], 0
    while i < len(args):
        a = str(args[i])
        if a in DROP:
            i += 1 + DROP[a]
            continue
        if a in ("--expert-swaps", "--temperature", "--tokens", "--max-new", "--serve"):
            i += 1 if a == "--serve" else 2
            continue
        out.append(a)
        i += 1
    return out + ["--expert-swaps", "0", "--temperature", "0", "--tokens", ",".join(map(str, ids)),
                  "--max-new", str(max_new)]


def first_difference(a: list[int], b: list[int]) -> int:
    """The first place two id lists differ (the shorter one's length when one is the start of the other), -1 if
    they are the same."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def parse_ids(stdout: str) -> list[int]:
    """The ids the one-shot mode prints: one line, comma-separated (the last such line)."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line and all(p.strip().lstrip("-").isdigit() for p in line.split(",")):
            return [int(p) for p in line.split(",")]
    return []


def cache_of(stderr: str) -> str:
    """What the engine's load says of its expert cache ("N of M routed experts in VRAM (G GiB)"), or "?": a model
    whose experts do not all fit gets as many as the free VRAM holds at the load, so two runs can differ in it."""
    for line in reversed(stderr.splitlines()):
        if "routed experts in VRAM" in line and "loaded in" in line:
            return line.split(";", 1)[-1].split(",")[0].strip()
    return "?"


def run_once(exe: str, cfg: dict, ids: list[int], max_new: int, timeout: float,
             freeze: bool = True) -> tuple[list[int], str]:
    import server   # serve/server.py: the engine's arguments and environment as the server gives them
    cmd = [exe] + regress_args(server.engine_args(cfg), ids, max_new)
    env = server.child_env(cfg)
    if freeze:
        env["STRATA_FREEZE_EXPERTS"] = "1"
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout, cwd=cfg.get("cwd") or None)
    got = parse_ids(r.stdout)
    if r.returncode != 0 or not got:
        tail = "\n".join((r.stderr or "").strip().splitlines()[-12:])
        raise RuntimeError(f"{Path(exe).name} exited {r.returncode} with {len(got)} ids:\n{tail}")
    return got, cache_of(r.stderr or "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("configs", nargs="+", help="strata-<model>.json run configs")
    ap.add_argument("--old", required=True, help="the engine before the change")
    ap.add_argument("--new", help="the engine after it (default: each config's own \"exe\")")
    ap.add_argument("--max-new", type=int, default=64, help="tokens each run writes (default 64)")
    ap.add_argument("--runs", type=int, default=2, help="runs of each engine (default 2)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--timeout", type=float, default=600, help="seconds a run may take (default 600)")
    ap.add_argument("--no-freeze", dest="freeze", action="store_false",
                    help="run without STRATA_FREEZE_EXPERTS (the experts' split and a window's way may then differ "
                         "between runs)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "serve"))
    import server
    bad = failed = 0
    for path in a.configs:
        cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        name = cfg.get("model_name") or Path(path).stem
        tok = server.load_tokenizer(Path(cfg.get("tokenizer") or ""))
        if tok is None:
            print(f"{name}: no tokenizer in {cfg.get('tokenizer')}")
            failed += 1
            continue
        ids = tok.encode(a.prompt)
        new = a.new or server.engine_exe(cfg)
        print(f"{name}: {len(ids)} prompt tokens, {a.max_new} to write, {a.runs} runs of each engine", flush=True)
        try:
            runs = {k: [run_once(exe, cfg, ids, a.max_new, a.timeout, a.freeze) for _ in range(a.runs)]
                    for k, exe in (("old", a.old), ("new", new))}
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"  a run failed: {e}")
            failed += 1
            continue
        outs = {k: [ids_ for ids_, _ in v] for k, v in runs.items()}
        caches = {k: [c for _, c in v] for k, v in runs.items()}
        for k in ("old", "new"):
            print(f"  {k} engine's expert cache, run by run: {' | '.join(caches[k])}")
        if len({c for v in caches.values() for c in v}) > 1:
            print("  (the runs' expert caches differ: experts computed on the card in one run and on the CPU in another"
                  " can part at a near tie)")
        for k in ("old", "new"):
            d = [first_difference(outs[k][0], o) for o in outs[k][1:]]
            same = all(x < 0 for x in d)
            print(f"  {k} engine repeats itself: {'yes' if same else 'NO, from token ' + str(min(x for x in d if x >= 0))}")
            bad += not same
        d = first_difference(outs["old"][0], outs["new"][0])
        print(f"  old == new: {'yes' if d < 0 else 'NO, from token ' + str(d)} ({len(outs['new'][0])} tokens)")
        if d >= 0:
            print(f"    old: {outs['old'][0][d:d + 12]}\n    new: {outs['new'][0][d:d + 12]}")
        bad += d >= 0
    return 2 if failed else 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
