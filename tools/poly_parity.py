"""strata-poly against llama.cpp on the same GGUF, token by token (laguna/parity.py's method, for any model both run).

    .venv/bin/python tools/poly_parity.py --gguf MODEL.gguf [--n 32] [--only short,code] [--ours-args "..."]

1. llama-server (engine-llama/ by default, the model as --fit places it) tokenizes each prompt (/tokenize, with the
   model's special tokens: the ids both engines read) and answers it greedily, with its 10 likeliest tokens a step
2. strata-poly (engine-poly/ by default) is made to write llama.cpp's answer (--follow) and prints its own 10 likeliest
   a step (--top-probs), with the expert split held still (STRATA_FREEZE_EXPERTS=1) and no guesses (--window 1)
3. per prompt: how many steps the two engines' first choices agree, the first step where they do not (and whether it
   is a near tie: either engine's two likeliest within 0.1), and KL(llama.cpp || strata-poly) over llama.cpp's 10

With --image (and --mmproj), a "picture" prompt too: llama-server reads the picture through its own mtmd, from its
media marker and the picture's file; strata-poly reads the same text with the picture's opening token, strata-vision's
rows of the picture (engine-poly/ by default, --vision) and its closing token, as mtmd puts them (Nemotron 3.5: <img>
and </img>; a Qwen-VL model such as Occamy: <|vision_start|> and <|vision_end|>), and the two prompts' token counts
are compared.

The server must not be running (both engines need the cards).  The report also goes to --out/parity-<time>.txt, the
raw steps to parity-<time>.json.  Exit 0 when every step's first choice agrees or parts at a near tie; 1 otherwise;
2 when a run fails.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOP = 10
PROMPTS = {
    "short": "The capital of France is",
    "code": 'def is_prime(n: int) -> bool:\n    """Return True if n is a prime number."""\n',
    "chat": "Explain in two sentences why the sky is blue.",
}


def doc_prompt(tokens_hint: int = 1500) -> str:
    """A longer prompt from the repository's docs (README.md), cut at about tokens_hint * 4 characters: the prompt
    batch's path (MMQ, the scan over many tokens)."""
    text = (ROOT / "README.md").read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
    return text[:tokens_hint * 4].rsplit("\n", 1)[0] + "\n\nIn one sentence, the text above describes"


def near_tie(top: list) -> bool:
    return len(top) >= 2 and top[0][1] - top[1][1] < 0.1


def compare(ref_tokens: list[int], ref_tops: list[list], ours: list[list]) -> dict:
    """The agreement of strata-poly's first choice with llama.cpp's token at each step."""
    n = min(len(ref_tokens), len(ours))
    agree, first, ties, kls = 0, None, 0, []
    for t in range(n):
        top1 = ours[t][0][0] if ours[t] else None
        if top1 == ref_tokens[t]:
            agree += 1
        else:
            if first is None:
                first = t
            lt = ref_tops[t] if t < len(ref_tops) else []
            if near_tie(lt) or near_tie(ours[t]):
                ties += 1
        if t < len(ref_tops) and ref_tops[t]:
            have = dict((i, q) for i, q in ours[t])
            kls.append(sum(q * math.log(q / max(have.get(i, 0.0), 1e-6)) for i, q in ref_tops[t] if q > 0))
    out = {"steps": n, "agree": agree, "first": first, "ties": ties,
           "kl_mean": sum(kls) / len(kls) if kls else None, "kl_max": max(kls) if kls else None}
    if first is not None:
        lt = ref_tops[first] if first < len(ref_tops) else []
        out["at_first"] = {"llama": [(i, round(q, 3)) for i, q in lt[:2]],
                           "ours": [(i, round(q, 3)) for i, q in ours[first][:2]],
                           "near_tie": near_tie(lt) or near_tie(ours[first])}
    return out


def parse_tops(stderr: str) -> list[list]:
    """strata-poly's --top-probs lines: "top id:p id:p ...", one a step."""
    tops = []
    for line in stderr.splitlines():
        if line.startswith("top "):
            tops.append([(int(i), float(q)) for i, q in (x.split(":") for x in line.split()[1:])])
    return tops


def lib_env(dirs: list[str]) -> dict:
    env = dict(os.environ)
    if dirs:
        var = "PATH" if os.name == "nt" else "LD_LIBRARY_PATH"
        env[var] = os.pathsep.join(dirs + ([env[var]] if env.get(var) else []))
    return env


PICTURE = ("Here is a picture:\n", "\nWhat does it show? Describe it in detail.")   # the text before it, and after it
# A picture's tokens: the two llama-server's mtmd puts around its rows (img_beg, img_end), and the one the model's
# template writes for it, whose place strata-poly's --image-pad fills with the rows.  Nemotron 3.5's first (mtmd's
# nemotron_omni), so its prompt is the one it always was; then Qwen-VL's (mtmd's qwen2vl, qwen25vl and qwen3vl;
# serve/server.py's IMAGE_PAD).  Qwen3.6's vocabulary (its tokenizer.json, read 2026-11-23) has these three and
# none of Nemotron's.
PICTURE_MARKS = (("<img>", "</img>", "<image>"), ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>"))


def picture_marks(tok) -> list | None:
    """The first of PICTURE_MARKS whose three are each one token of the model's (tok(text) -> its ids): their ids
    [[begin], [end], [pad]]; None when no set is."""
    for marks in PICTURE_MARKS:
        ids = [tok(m) for m in marks]
        if all(len(x) == 1 for x in ids):
            return ids
    return None


def encode_picture(vision: str, mmproj: str, gguf: str, image: str, out: Path, env: dict) -> int:
    """The picture's rows from strata-vision (one process, on the card, at the encoder file's own limits): its rows."""
    p = subprocess.Popen([vision, "--mmproj", mmproj, "--model", gguf, "--gpu"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        ready = p.stdout.readline().strip()
        if not ready.startswith("READY"):
            raise RuntimeError(f"strata-vision did not start: {ready or p.stderr.read()[-800:]}")
        p.stdin.write(f"ENC {Path(image).resolve()} {out.resolve()}\n")
        p.stdin.flush()
        reply = p.stdout.readline().strip()
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=120)
    finally:
        if p.poll() is None:
            p.kill()
    if not reply.startswith("OK "):
        raise RuntimeError(f"strata-vision: {reply}")
    return int(reply.split()[1])


def post(base: str, path: str, body: dict, timeout: float = 3600) -> dict:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gguf", required=True, help="the model's GGUF (the first file of a split one)")
    ap.add_argument("--engine", default=str(ROOT / "engine-poly" / "strata-poly"))
    ap.add_argument("--llama-server", default=str(ROOT / "engine-llama" / "llama-server"))
    ap.add_argument("--n", type=int, default=32, help="tokens each answer has (default 32)")
    ap.add_argument("--context", type=int, default=8192)
    ap.add_argument("--only", default="", help="comma-separated prompt names: short, code, chat, doc")
    ap.add_argument("--port", type=int, default=8093)
    ap.add_argument("--ours-args", default="", help="more strata-poly options, as one string")
    ap.add_argument("--llama-args", default="", help="more llama-server options, as one string")
    ap.add_argument("--lib-dirs", default="", help="library folders for both engines, a path list (default: the "
                    "toolkit's, from engine-poly/BUILD.json)")
    ap.add_argument("--image", help="a picture: adds the \"picture\" prompt (needs --mmproj)")
    ap.add_argument("--mmproj", help="the model's image encoder file, for both engines")
    ap.add_argument("--vision", default=str(ROOT / "engine-poly" / ("strata-vision.exe" if os.name == "nt" else
                                                                    "strata-vision")))
    ap.add_argument("--out", default=str(ROOT / "parity-test"))
    # an option's value that starts with -- is taken by argparse for an option: join it to its flag
    argv = list(sys.argv[1:] if argv is None else argv)
    for i in range(len(argv) - 1):
        if argv[i] in ("--ours-args", "--llama-args"):
            argv[i:i + 2] = [argv[i] + "=" + argv[i + 1]]
            break
    a = ap.parse_args(argv)
    if a.image and not a.mmproj:
        ap.error("--image needs --mmproj")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = open(out / f"parity-{stamp}.txt", "w", encoding="utf-8")

    def say(*parts):
        line = " ".join(str(x) for x in parts)
        print(line, flush=True)
        report.write(line + "\n")
        report.flush()

    dirs = [d for d in a.lib_dirs.split(os.pathsep) if d]
    if not dirs:
        try:
            dirs = json.loads((ROOT / "engine-poly" / "BUILD.json").read_text(encoding="utf-8")).get("cuda_dirs") or []
        except (OSError, ValueError):
            dirs = []
    env = lib_env(dirs)
    prompts = dict(PROMPTS, doc=doc_prompt())
    if a.image:
        prompts["picture"] = PICTURE
    if a.only:
        prompts = {k: v for k, v in prompts.items() if k in a.only.split(",")}
    say("== strata-poly:", a.engine, "| options beside the defaults:", a.ours_args or "(none)")
    say("== the model:", a.gguf)

    # 0) the picture's rows, before llama-server takes the cards (its media marker is its own each start: read below)
    n_rows, rows_file = 0, out / f"picture-{stamp}.sve"
    if a.image:
        try:
            n_rows = encode_picture(a.vision, a.mmproj, a.gguf, a.image, rows_file, env)
        except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
            say(f"   the picture could not be encoded: {e}")
            return 2
        say(f"== strata-vision: {a.image} is {n_rows} rows")

    # 1) llama.cpp
    cmd = [a.llama_server, "-m", a.gguf, "-c", str(a.context), "-np", "1", "--host", "127.0.0.1", "--port",
           str(a.port), "--no-webui"] + (["--mmproj", a.mmproj] if a.mmproj else []) + a.llama_args.split()
    say("== llama-server:", " ".join(cmd))
    slog = open(out / f"llama-server-{stamp}.log", "w", encoding="utf-8")
    srv = subprocess.Popen(cmd, stdout=slog, stderr=subprocess.STDOUT, env=env)
    base = f"http://127.0.0.1:{a.port}"
    ids, ref, pictures = {}, {}, {}
    try:
        t0 = time.time()
        while True:
            if srv.poll() is not None:
                say(f"llama-server stopped (code {srv.returncode}): see {slog.name}")
                return 2
            try:
                with urllib.request.urlopen(base + "/health", timeout=5) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                pass
            if time.time() - t0 > 900:
                say("llama-server did not come up in 15 minutes")
                return 2
            time.sleep(2)
        say(f"   up in {time.time() - t0:.0f} s")
        marker = None
        for k, text in prompts.items():
            if k == "picture":
                # the same tokens for strata-poly: the text, the picture's opening token, a pad a row of the
                # picture, its closing token, the text
                with urllib.request.urlopen(base + "/props", timeout=30) as r:
                    marker = json.loads(r.read()).get("media_marker")
                tok = lambda t, first: post(base, "/tokenize", {"content": t, "add_special": first,   # noqa: E731
                                                                "parse_special": True})["tokens"]
                marks = picture_marks(lambda t: tok(t, False))
                if not (marker and marks):
                    say("   the picture prompt needs llama-server's media marker and, as single tokens, one of: "
                        + "; ".join(" ".join(m) for m in PICTURE_MARKS))
                    return 2
                img, img_end, pad = marks
                ids[k] = tok(text[0], True) + img + pad * n_rows + img_end + tok(text[1], False)
                pictures[k] = (rows_file, pad[0], n_rows)
                picture = base64.b64encode(Path(a.image).read_bytes()).decode("ascii")
                prompt = {"prompt_string": text[0] + marker + text[1], "multimodal_data": [picture]}
            else:
                ids[k] = post(base, "/tokenize", {"content": text, "add_special": True,
                                                  "parse_special": True})["tokens"]
                prompt = ids[k]
            body = {"prompt": prompt, "n_predict": a.n, "temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0,
                    "repeat_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0, "dry_multiplier": 0.0,
                    "n_probs": TOP, "post_sampling_probs": False, "cache_prompt": False, "return_tokens": True}
            t1 = time.time()
            res = post(base, "/completion", body)
            steps = []
            for e in res.get("completion_probabilities") or []:
                tops = e.get("top_logprobs") or e.get("top_probs") or []
                steps.append([(t["id"], math.exp(t["logprob"]) if "logprob" in t else t["prob"]) for t in tops])
            toks = res.get("tokens") or [s[0][0] for s in steps if s]
            ref[k] = {"tokens": toks, "tops": steps}
            read = res.get("tokens_evaluated")
            say(f"   llama.cpp {k}: {len(ids[k])} prompt tokens, {len(toks)} written in {time.time() - t1:.0f} s"
                + (f"; the picture's rows: {pictures[k][2]}; llama-server read {read} prompt tokens, strata-poly's "
                   f"prompt has {len(ids[k])} ({'the same' if read == len(ids[k]) else 'NOT the same'})"
                   if k in pictures else ""))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
        slog.close()

    # 2) strata-poly, made to write llama.cpp's answers
    results, bad, failed = {}, 0, 0
    for k in prompts:
        follow = ref[k]["tokens"]
        cmd = [a.engine, "--model", a.gguf, "--tokens", ",".join(map(str, ids[k])), "--follow",
               ",".join(map(str, follow)), "--max-new", str(len(follow)), "--top-probs", str(TOP), "--max-context",
               str(a.context), "--layer-split", "auto", "--window", "1"] + a.ours_args.split()
        if k in pictures:
            cmd += ["--images", str(pictures[k][0]), "--image-pad", str(pictures[k][1])]
        t1 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, env=dict(env, STRATA_FREEZE_EXPERTS="1"))
        (out / f"strata-{stamp}.log").open("a", encoding="utf-8").write(f"== {k}\n{r.stderr}")
        if r.returncode != 0:
            say(f"   strata-poly {k}: failed ({r.returncode}):")
            say("\n".join("     " + x for x in r.stderr.strip().splitlines()[-15:]))
            failed += 1
            continue
        ours = parse_tops(r.stderr)
        c = compare(follow, ref[k]["tops"], ours)
        results[k] = {"ours": ours, "cmp": c}
        line = f"   {k}: {c['agree']} of {c['steps']} first choices agree ({time.time() - t1:.0f} s)"
        if c["first"] is not None:
            f = c["at_first"]
            line += (f"; first differs at step {c['first']}: llama.cpp {f['llama']} strata-poly {f['ours']}"
                     f" ({'near tie' if f['near_tie'] else 'NOT a near tie'}); {c['steps'] - c['agree'] - c['ties']}"
                     " differences not at near ties")
        if c["kl_mean"] is not None:
            line += f"; KL mean {c['kl_mean']:.2e}, max {c['kl_max']:.2e}"
        say(line)
        bad += c["steps"] - c["agree"] - c["ties"] > 0
    json.dump({"prompts": ids, "ref": ref, "ours": results, "ours_args": a.ours_args},
              open(out / f"parity-{stamp}.json", "w", encoding="utf-8"))
    say(f"== {out / f'parity-{stamp}.txt'}")
    return 2 if failed else 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
