"""Does a new image encoder (strata-vision) give a model's pictures the rows the old one gave?

For a run config with a "vision" section (strata-<model>.json), each picture goes through both encoders, one after
the other, with the config's encoder file (mmproj), model, device and token limit, as the server starts the encoder:
the old one (a copy made before an update) and the new one (the config's own, or --new).  Then their records are
compared: the rows and the grid, and the values (the largest difference, and how many are the same to the bit).

    .venv/bin/python tools/vision_compare.py --old ~/strata-vision-old strata-gemma4-qat-ud-q4_k_xl.json \\
        --image third_party/llama.cpp-08c252e/tools/mtmd/test-1.jpeg

The server must not be running when the encoder is on the card.  Exit 0: every picture's rows are the same to the bit;
1: they differ (printed); 2: an encoder failed.
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAGIC = 0x31455653   # 'SVE1'


def read_records(path: Path) -> list[tuple[int, int, int, array]]:
    """strata-vision's output: records of int32 {magic, n, nx, ny, n_embd} and n * n_embd float32."""
    data = path.read_bytes()
    out, at = [], 0
    while at < len(data):
        magic, n, nx, ny, n_embd = struct.unpack_from("<5i", data, at)
        if magic != MAGIC:
            raise ValueError(f"{path}: not strata-vision's records")
        at += 20
        rows = array("f")
        rows.frombytes(data[at:at + n * n_embd * 4])
        at += n * n_embd * 4
        out.append((n, nx, ny, rows))
    return out


def compare_rows(a: array, b: array) -> dict:
    """The largest absolute difference, the largest value, and the share of values the same to the bit."""
    if len(a) != len(b):
        return {"same_length": False}
    worst = top = 0.0
    same = 0
    for x, y in zip(a, b):
        d = abs(x - y)
        if d > worst or d != d:
            worst = d
        top = max(top, abs(x))
        same += x == y or (x != x and y != y)
    return {"same_length": True, "max_abs": worst, "max_value": top, "same_share": same / len(a) if a else 1.0}


def encoder_cmd(exe: str, vision: dict) -> list[str]:
    cmd = [exe, "--mmproj", vision["mmproj"], "--model", vision["model"]]
    if vision.get("gpu"):
        cmd.append("--gpu")
    if vision.get("threads"):
        cmd += ["--threads", str(vision["threads"])]
    if vision.get("max_tokens"):
        cmd += ["--max-tokens", str(vision["max_tokens"])]
    return cmd


def encode(exe: str, vision: dict, env: dict, images: list[Path], out_dir: Path, tag: str) -> list[tuple[str, Path]]:
    """Every picture through one encoder process: (its "OK ..." line, the records file)."""
    p = subprocess.Popen(encoder_cmd(exe, vision), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, env=env)
    try:
        line = p.stdout.readline().strip()
        if not line.startswith("READY"):
            raise RuntimeError(f"{exe} did not start: {line or p.stderr.read()[-800:]}")
        got = []
        for k, img in enumerate(images):
            out = out_dir / f"{tag}-{k}.sve"
            p.stdin.write(f"ENC {img} {out}\n")
            p.stdin.flush()
            reply = p.stdout.readline().strip()
            if not reply.startswith("OK"):
                raise RuntimeError(f"{exe} on {img}: {reply}")
            got.append((reply, out))
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=120)
        return got
    finally:
        if p.poll() is None:
            p.kill()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("config", help="a strata-<model>.json with a \"vision\" section")
    ap.add_argument("--old", required=True, help="the encoder before the change")
    ap.add_argument("--new", help="the encoder after it (default: the config's own)")
    ap.add_argument("--image", action="append", required=True, help="a picture (more than one: --image again)")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(ROOT / "serve"))
    import server
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8-sig"))
    vision = cfg.get("vision")
    if not isinstance(vision, dict):
        print(f"{a.config} has no \"vision\" section")
        return 2
    env = server.child_env(cfg)
    images = [Path(x).resolve() for x in a.image]
    with tempfile.TemporaryDirectory() as d:
        try:
            old = encode(a.old, vision, env, images, Path(d), "old")
            new = encode(a.new or vision["exe"], vision, env, images, Path(d), "new")
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            print(f"an encoder failed: {e}")
            return 2
        differ = 0
        for img, (o_line, o_path), (n_line, n_path) in zip(images, old, new):
            ro, rn = read_records(o_path), read_records(n_path)
            print(f"{img.name}: old {o_line} | new {n_line}")
            if [(n, nx, ny) for n, nx, ny, _ in ro] != [(n, nx, ny) for n, nx, ny, _ in rn]:
                print(f"  the rows differ in number or grid: old {[(n, nx, ny) for n, nx, ny, _ in ro]}, "
                      f"new {[(n, nx, ny) for n, nx, ny, _ in rn]}")
                differ += 1
                continue
            for k, ((n, nx, ny, a_rows), (_, _, _, b_rows)) in enumerate(zip(ro, rn)):
                c = compare_rows(a_rows, b_rows)
                print(f"  record {k + 1}: {n} rows ({nx} x {ny}); the same to the bit: {100 * c['same_share']:.2f}%; "
                      f"largest difference {c['max_abs']:.3g} (values up to {c['max_value']:.3g})")
                differ += c["same_share"] < 1.0
    return 1 if differ else 0


if __name__ == "__main__":
    sys.exit(main())
