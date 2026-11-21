"""tools/llama_server_check.py - a llama-server model through Puddle against llama-server alone, greedy, on one
question with a picture (or none).

Puddle renders the chat template, tokenizes the text and sends llama-server the prompt (as token ids, or as text with
llama-server's media marker when there is a picture); llama-server alone does all of that itself through its own
/v1/chat/completions.  The check compares what the model was given - the prompt-token counts (Puddle plans a
picture's tokens by serve/media_tokens.py) and both prompts as llama-server read them (its own detokenization:
Puddle's from its last event, through STRATA_LLAMA_PROMPT_FILE; llama-server alone's from a "verbose" reply) - and
what it wrote, greedy.

The two answers can part at a near tie even with the same prompt: llama-server's chat endpoints end a prompt batch
where a user message starts (for their checkpoints of the Mamba state: server-context.cpp's message_spans, at
08c252e22), and /completion, which Puddle uses, has no such breaks, so the same prompt is read in other batches and
rounds a little differently.  So the check also replays Puddle's own request (STRATA_LLAMA_REQUEST_FILE) on
llama-server alone's /completion: that one must write Puddle's answer exactly.

The model must not be running: the check starts Puddle with the run config (on 127.0.0.1, its own port), asks, stops
it, then starts llama-server with the config's own command and environment (the cards in the same order), asks,
stops it, and compares.  One model in VRAM at a time.

    python tools/llama_server_check.py --config strata-nemotron-llama-bolt-24gb.json --image picture.jpg
        [--prompt "What does this picture show? Two sentences."] [--max-tokens 600]

Exit code 0: the same prompt, and Puddle's answer from llama-server alone's /completion (the chat endpoint's answer
the same or not); 1: a difference in the prompt or in the replay (the first one is printed); 2: something did not
start or answer.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def error_words(e: Exception) -> str:
    """An HTTP error with the server's own message (its JSON "error" body), else the error as it is."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = json.loads(e.read() or b"{}")
            msg = (body.get("error") or {}).get("message") if isinstance(body.get("error"), dict) else body.get("error")
            if msg:
                return f"HTTP {e.code}: {msg}"
        except (ValueError, OSError, AttributeError):
            pass
    return str(e)


def get_json(url: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def post(url: str, body: dict, key: str | None = None, timeout: float = 900.0) -> dict:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {key}"} if key else {})}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_up(url: str, proc: subprocess.Popen, what: str, timeout: float = 900.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited before it answered (code {proc.returncode})")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"{what} did not answer {url} within {timeout:.0f} s")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)


def stream_text(url: str, body: dict, timeout: float = 900.0) -> str:
    """The text of a streamed /completion reply (its `data:` events' content, in order)."""
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if line.startswith(b"data: "):
                ev = json.loads(line[6:])
                if "prompt_progress" not in ev:
                    out.append(ev.get("content") or "")
                if ev.get("stop"):
                    break
    return "".join(out)


def first_difference(a: str, b: str, around: int = 40) -> str | None:
    """None when equal, else where they part, with some text of each."""
    if a == b:
        return None
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return (f"at character {i}:\n    Puddle:       {a[max(0, i - around):i + around]!r}\n"
            f"    llama-server: {b[max(0, i - around):i + around]!r}")


def answer(reply: dict) -> tuple[str, str, int | None]:
    msg = (reply.get("choices") or [{}])[0].get("message") or {}
    return (msg.get("reasoning_content") or ""), (msg.get("content") or ""), (reply.get("usage") or {}).get(
        "prompt_tokens")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", required=True, help="the model's run config (\"engine\": \"llama-server\")")
    ap.add_argument("--image", help="a picture to ask about (leave out for text alone)")
    ap.add_argument("--prompt", default="What does this picture show? Two sentences.")
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--log", default="llama_server_check.log", help="both servers' output (default: %(default)s)")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    if cfg.get("engine") != "llama-server":
        print(f"{cfg_path.name} is not a llama-server model's run config")
        return 2
    content: list | str = a.prompt
    if a.image:
        data = Path(a.image).read_bytes()
        kind = "png" if data[:4] == b"\x89PNG" else "jpeg"
        url = f"data:image/{kind};base64," + base64.b64encode(data).decode()
        content = [{"type": "text", "text": a.prompt}, {"type": "image_url", "image_url": {"url": url}}]
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": a.max_tokens, "temperature": 0,
            "stream": False}
    log = open(a.log, "w", encoding="utf-8")

    # 1. through Puddle (the run config's own settings; greedy by the request)
    port = free_port()
    prompt_file = Path(a.log).with_suffix(".puddle-prompt.txt")
    request_file = Path(a.log).with_suffix(".puddle-request.json")
    for f in (prompt_file, request_file):
        f.unlink(missing_ok=True)
    puddle = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config",
                               str(cfg_path), "--host", "127.0.0.1", "--port", str(port)],
                              cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                              env={**{k: v for k, v in os.environ.items() if k != "STRATA_API_KEY"},
                                   "STRATA_LLAMA_PROMPT_FILE": str(prompt_file.resolve()),
                                   "STRATA_LLAMA_REQUEST_FILE": str(request_file.resolve())})
    try:
        print("starting Puddle with the model ...", flush=True)
        wait_up(f"http://127.0.0.1:{port}/health", puddle, "Puddle")
        key = str(cfg.get("api_key") or "").split(",")[0].strip() or None   # the config's own key, if it has one
        mine = post(f"http://127.0.0.1:{port}/v1/chat/completions", body, key)
    except (RuntimeError, OSError, urllib.error.URLError, ValueError) as e:
        print(f"Puddle: {error_words(e)} (its output: {a.log})")
        return 2
    finally:
        stop(puddle)
    time.sleep(5)                                       # the cards' memory comes back with the processes

    # 2. llama-server alone: the config's command and environment (serve/server.py's child_env: the same cards, in
    # the same order, and the same libraries)
    from serve import server as S
    env = S.child_env(cfg)
    port = free_port()
    llama = subprocess.Popen([S.engine_exe(cfg), *[str(x) for x in cfg["args"]], "--host", "127.0.0.1", "--port",
                              str(port), "--no-ui"], cwd=cfg.get("cwd") or str(ROOT), stdout=log,
                             stderr=subprocess.STDOUT, env=env)
    try:
        print("starting llama-server alone ...", flush=True)
        wait_up(f"http://127.0.0.1:{port}/health", llama, "llama-server")
        theirs = post(f"http://127.0.0.1:{port}/v1/chat/completions", {**body, "verbose": True})
        # Puddle's own request, as it sent it, on a fresh slot (the chat request's cache would be reused otherwise)
        replay = json.loads(request_file.read_text(encoding="utf-8")) if request_file.is_file() else None
        if replay is not None:
            replay["cache_prompt"] = False
            prompt = replay.get("prompt")
            if isinstance(prompt, dict) and isinstance(prompt.get("prompt_string"), str):
                # llama-server makes a new media marker at each start: this one's goes where Puddle's was
                marker = get_json(f"http://127.0.0.1:{port}/props").get("media_marker") or ""
                prompt["prompt_string"] = re.sub(r"<__media_[A-Za-z0-9]*__>", lambda m: marker,
                                                 prompt["prompt_string"])
            replayed = stream_text(f"http://127.0.0.1:{port}/completion", replay)
    except (RuntimeError, OSError, urllib.error.URLError, ValueError) as e:
        print(f"llama-server: {error_words(e)} (its output: {a.log})")
        return 2
    finally:
        stop(llama)
        log.close()

    r1, c1, n1 = answer(mine)
    r2, c2, n2 = answer(theirs)
    print(f"prompt tokens: Puddle planned {n1}, llama-server counted {n2}" +
          (" (the same)" if n1 == n2 else f" (off by {n1 - n2})" if n1 is not None and n2 is not None else ""))
    p1 = prompt_file.read_text(encoding="utf-8") if prompt_file.is_file() else None
    p2 = (theirs.get("__verbose") or {}).get("prompt")
    if p1 is None or p2 is None:
        print("prompts: not both known (" + ("Puddle's missing" if p1 is None else "llama-server's missing") + ")")
    else:
        d = first_difference(p1, p2)
        print(f"prompts as llama-server read them: the same ({len(p1)} characters)" if d is None else
              f"prompts as llama-server read them: differ {d}")
        Path(a.log).with_suffix(".llama-prompt.txt").write_text(p2, encoding="utf-8")
    prompt_same = p1 is not None and p1 == p2 and n1 == n2
    replay_same = None
    if replay is None:
        print("replay: Puddle's request was not written")
    else:
        replay_same = r1.strip() in replayed and c1.strip() in replayed
        print("Puddle's request on llama-server alone's /completion: " +
              ("the same thinking and answer as through Puddle" if replay_same else
               "differs from Puddle's answer " + (first_difference(r1.strip(), replayed[:len(r1.strip())]) or "")))
    same = True
    for what, x, y in (("thinking", r1, r2), ("answer", c1, c2)):
        d = first_difference(x, y)
        if d is None:
            print(f"{what}: the same ({len(x)} characters)")
        else:
            same = False
            print(f"{what}: differs {d}")
    if not same and first_difference(r1 + c1, r2 + c2) is None:
        print("(the whole text is the same: only where the thinking ends differs)")
    if not same and prompt_same and replay_same:
        print("(the chat endpoint's answer parts from it with the same prompt: it reads the prompt in other batches, "
              "ending one where a user message starts)")
    print("\nPuddle's answer:\n" + c1)
    return 0 if prompt_same and (same or replay_same) else 1


if __name__ == "__main__":
    raise SystemExit(main())
