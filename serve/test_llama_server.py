"""serve/server.py's LlamaServerEngine (a run config with "engine": "llama-server") against a fake llama-server
(serve/fixtures/fake_llama_server.py): no GPU, no model, no llama.cpp build.

    python -m unittest serve.test_llama_server
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

from serve import models as registry
from serve import server as S
from serve.frontend import ChatTemplate
from serve.test_models import gguf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402

FAKE = str(ROOT / "serve" / "fixtures" / "fake_llama_server.py")
# the test vocabulary: the 256 bytes, four longer pieces, and the special tokens
PIECES = [bytes([b]) for b in range(256)] + [b"\xf0\x9f", b"\x9f\x98", b"Hello", b" world"]
SPECIALS = [("<|im_start|>", 3), ("<|im_end|>", 3), ("</s>", 3), ("<think>", 4)]
F09F, X9F98, HELLO, WORLD = 256, 257, 258, 259
IM_START, IM_END, EOS_S, THINK = 260, 261, 262, 263


def tokenizer() -> ST.Tokenizer:
    tokens = ["".join(ST.BYTE_TO_UNICODE[b] for b in p) for p in PIECES] + [s for s, _ in SPECIALS]
    return ST.Tokenizer(tokens, [], [1] * len(PIECES) + [t for _, t in SPECIALS], pre="pixtral")


class Fake:
    """A fake llama-server's files: its vocabulary, its replies, the record of its requests and its log."""

    def __init__(self, test: unittest.TestCase, replies: list[list[int]], *extra: str, ctx: int = 4096):
        d = tempfile.TemporaryDirectory()
        test.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        (self.dir / "vocab.json").write_text(json.dumps({"tokens": [p.hex() for p in PIECES] + [s.encode().hex()
                                                                                              for s, _ in SPECIALS],
                                                          "special": [IM_START, IM_END, EOS_S]}))
        (self.dir / "reply.json").write_text(json.dumps(replies))
        self.record, self.log = self.dir / "requests.jsonl", self.dir / "llama.log"
        self.args = [FAKE, "-m", str(self.dir / "bolt.gguf"), "-c", str(ctx), "-np", "1",
                     "--fake-vocab", str(self.dir / "vocab.json"), "--fake-reply", str(self.dir / "reply.json"),
                     "--fake-eog", f"{IM_END},{EOS_S}", "--fake-ctx", str(ctx), "--fake-record", str(self.record),
                     *extra]

    def engine(self, test: unittest.TestCase, lazy: bool = False, quiet: bool = True) -> S.LlamaServerEngine:
        out = io.StringIO()
        with contextlib.redirect_stdout(out) if quiet else contextlib.nullcontext():
            e = S.LlamaServerEngine(sys.executable, self.args, log=str(self.log), lazy=lazy, tokenizer=tokenizer())
        self.said = out.getvalue()
        e.eos_ids = frozenset({IM_END})                  # what Service sets: <|im_end|>
        test.addCleanup(e.close)
        return e

    def requests(self) -> list[dict]:
        if not self.record.exists():
            return []
        return [json.loads(x) for x in self.record.read_text(encoding="utf-8").splitlines() if x.strip()]


def run(engine, ids, max_new=64, sampling=None, cancel=None, stop_at=None):
    """The tokens as Service.run takes them: heartbeats left out, and the generator closed at a stop id."""
    out = []
    gen = engine.generate(ids, max_new, sampling or {}, cancel or threading.Event())
    try:
        for t in gen:
            if t is None:
                continue
            out.append(t)
            if t in (stop_at or engine.eos_ids):
                break
    finally:
        gen.close()
    return out


class Lifecycle(unittest.TestCase):
    def test_start_waits_for_the_model_and_reads_the_context(self):
        fake = Fake(self, [[HELLO, IM_END]], "--fake-load-s", "0.5", ctx=8192)
        e = fake.engine(self)
        self.assertTrue(e.alive())
        self.assertEqual((e.max_context, e.known_ctx), (8192, 8192))
        self.assertEqual(e.info["version"], "llama-server b0-fake")
        self.assertEqual(e.model_path, str(fake.dir / "bolt.gguf"))
        started = fake.log.read_text(encoding="utf-8")
        self.assertIn("--host 127.0.0.1 --port", started)
        self.assertIn("--no-ui", started)
        self.assertNotIn(e.key, started)                 # the key is in its environment, not on its command line
        self.assertIn("llama-server is ready (context 8192 tokens", fake.said)

    def test_only_this_server_has_the_key(self):
        fake = Fake(self, [[HELLO, IM_END]])
        e = fake.engine(self)
        run(e, [1, 2, 3])
        self.assertEqual([r["keyed"] for r in fake.requests()], [True])
        e.key = "wrong"
        with self.assertRaisesRegex(ValueError, "Invalid API Key"):
            run(e, [1, 2, 3])

    def test_lazy_unload_and_restart(self):
        fake = Fake(self, [[HELLO, IM_END]], ctx=6000)
        e = fake.engine(self, lazy=True)
        self.assertFalse(e.alive())
        self.assertEqual((e.max_context, e.known_ctx), (0, 6000))   # the config's -c until it has started
        with self.assertRaises(S.EngineDied):
            run(e, [1])
        with contextlib.redirect_stdout(io.StringIO()):
            e.restart()
        port = e.port
        self.assertTrue(e.alive())
        self.assertEqual(run(e, [1]), [HELLO, IM_END])
        e.unload()
        self.assertFalse(e.alive())
        self.assertTrue(e.unloaded)
        with contextlib.redirect_stdout(io.StringIO()):
            e.restart()
        self.assertTrue(e.alive())
        self.assertEqual(run(e, [1]), [HELLO, IM_END])
        self.assertTrue(port > 0 and e.port > 0)

    def test_a_start_that_fails_says_why(self):
        fake = Fake(self, [[HELLO]], "--fake-exit-at-start")
        with self.assertRaises(S.EngineDied) as cm:
            fake.engine(self)
        self.assertIn("llama-server exited before it was ready", str(cm.exception))
        self.assertIn("the model could not be loaded", str(cm.exception))

    def test_a_port_taken_meanwhile_is_tried_again_on_another(self):
        fake = Fake(self, [[HELLO, IM_END]], "--fake-bind-fail-once", "")
        fake.args[-1] = str(fake.dir / "bound-once")
        e = fake.engine(self)
        self.assertTrue(e.alive())
        self.assertIn("starting it on another one", fake.said)

    def test_the_config_cannot_set_its_address_or_key(self):
        self.assertIsNone(S.llama_server_args_problem(["-m", "x.gguf", "-c", "8192"]))
        for bad in (["--port", "8080"], ["--host=0.0.0.0"], ["--api-key", "k"], ["--api-key-file", "f"]):
            self.assertIn("must not set", S.llama_server_args_problem(["-m", "x.gguf", *bad]))
        for cfg in ({"engine": "llama-server", "exe": "x", "args": ["--port", "1"]},
                    {"engine": "llama-server", "exe": "x", "args": [], "vision": {"mmproj": "m"}},
                    {"engine": "llama-server", "exe": "x", "args": [], "parallel": 2},
                    {"engine": "llamacpp", "exe": "x", "args": []}):
            with self.assertRaises(SystemExit), contextlib.redirect_stdout(io.StringIO()):
                S.start_model(cfg, tokenizer(), lazy=True)

    def test_start_model_makes_one(self):
        fake = Fake(self, [[HELLO, IM_END]])
        cfg = {"engine": "llama-server", "exe": sys.executable, "args": fake.args, "log": str(fake.log),
               "engine_silence_s": 42}
        with contextlib.redirect_stdout(io.StringIO()):
            engine, vision, sampling, groups = S.start_model(cfg, tokenizer(), lazy=True)
        self.addCleanup(engine.close)
        self.assertIsInstance(engine, S.LlamaServerEngine)
        self.assertEqual((vision, groups, engine.silence_s), (None, [], 42.0))


class Requests(unittest.TestCase):
    def test_the_prompt_goes_as_ids_and_the_figures_are_llama_servers(self):
        fake = Fake(self, [[HELLO, WORLD, IM_END]])
        e = fake.engine(self)
        self.assertEqual(run(e, [5, 6, 7], max_new=32), [HELLO, WORLD, IM_END])
        body = fake.requests()[0]["body"]
        self.assertEqual(body["prompt"], [5, 6, 7])
        self.assertEqual(body["n_predict"], 32)
        self.assertTrue(body["stream"] and body["return_tokens"] and body["return_progress"] and body["cache_prompt"])
        # Puddle stopped at <|im_end|>: the last event was still read, so the figures are llama-server's
        self.assertEqual(e.last, {"generated": 3, "prompt_tokens": 3, "prompt_ms": 12.5, "decode_ms": 40.0,
                                  "finish": "stop", "reused": 0, "prompt_read": 3})
        self.assertEqual(e.progress, (3, 3))

    def test_the_prompt_llama_server_read_can_be_written_out(self):
        fake = Fake(self, [[HELLO, IM_END]])
        e = fake.engine(self)
        out = fake.dir / "prompt.txt"
        with unittest.mock.patch.dict(S.os.environ, {"STRATA_LLAMA_PROMPT_FILE": str(out)}):
            run(e, [5, 6, 7])
        self.assertEqual(out.read_text(encoding="utf-8"), "the prompt, 3 tokens")
        out.unlink()
        run(e, [5, 6, 7])                                  # off by default
        self.assertFalse(out.exists())

    def test_a_reply_cut_by_max_tokens_ends_as_length(self):
        fake = Fake(self, [[HELLO] * 10])
        e = fake.engine(self)
        self.assertEqual(run(e, [1], max_new=4), [HELLO] * 4)
        self.assertEqual(e.last["finish"], "length")

    def test_the_next_turn_reuses_the_cached_start(self):
        fake = Fake(self, [[HELLO, IM_END]])
        e = fake.engine(self)
        run(e, list(range(1, 21)))
        run(e, list(range(1, 21)) + [HELLO, IM_END, 9, 9])
        self.assertEqual((e.last["reused"], e.last["prompt_read"]), (22, 2))

    def test_sampling_means_what_it_means_with_strata_poly(self):
        f = S.LlamaServerEngine.sampling_fields
        greedy = f({})
        self.assertEqual({k: greedy[k] for k in ("temperature", "top_k", "top_p", "min_p", "repeat_penalty",
                                                 "frequency_penalty", "presence_penalty", "repeat_last_n", "mirostat")},
                         {"temperature": 0.0, "top_k": 20, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                          "frequency_penalty": 0.0, "presence_penalty": 0.0, "repeat_last_n": 0, "mirostat": 0})
        self.assertEqual(greedy["samplers"], ["penalties", "top_k", "top_p", "min_p", "temperature"])
        self.assertNotIn("seed", greedy)
        s = f({"temperature": 0.7, "top_k": 0, "top_p": 0.95, "min_p": 0.1, "presence_penalty": 1.5, "seed": 7})
        self.assertEqual((s["temperature"], s["top_k"], s["top_p"], s["min_p"], s["presence_penalty"],
                          s["repeat_last_n"], s["seed"]), (0.7, 64, 0.95, 0.1, 1.5, 64, 7))
        self.assertEqual(f({"top_k": 100})["top_k"], 64)
        self.assertEqual(f({"top_k": 40, "repetition_penalty": 1.1, "penalty_last_n": 256})["repeat_last_n"], 256)
        self.assertEqual(f({"temperature": -1, "top_p": 1.5, "min_p": 0, "seed": 0, "top_k": True})["top_k"], 20)
        fake = Fake(self, [[HELLO, IM_END]])
        e = fake.engine(self)
        run(e, [1], sampling={"temperature": 0.6, "top_p": 0.95, "stop": ["x"], "image_pad": 3})
        body = fake.requests()[0]["body"]
        self.assertEqual((body["temperature"], body["top_p"], body["top_k"]), (0.6, 0.95, 20))
        self.assertNotIn("stop", body)                   # Puddle cuts at its stop strings itself
        self.assertNotIn("image_pad", body)

    def test_a_character_split_over_tokens_comes_whole(self):
        tok = tokenizer()
        fake = Fake(self, [[0xC3, 0xA9, HELLO, IM_END], [F09F, 0x98, 0x80, IM_END], [0xF0, X9F98, 0x80, IM_END]])
        e = fake.engine(self)
        self.assertEqual(run(e, [1]), [0xC3, 0xA9, HELLO, IM_END])          # one token unsent: exact
        self.assertEqual(run(e, [1]), [F09F, 0x98, 0x80, IM_END])           # two: the longest pieces first
        got = run(e, [1])                                # two sent as other ones: the same text, as many tokens
        self.assertEqual(tok.decode(got[:-1]), "\U0001f600")
        self.assertEqual(len(got), 4)
        self.assertEqual(e.rebuilt, 1 + 2 + 2)

    def test_llama_cpps_own_end_of_generation_ends_the_reply_as_a_stop(self):
        fake = Fake(self, [[HELLO, EOS_S], [THINK, HELLO, IM_END]])
        e = fake.engine(self)
        self.assertEqual(run(e, [1]), [HELLO, IM_END])     # </s>: Puddle's end of turn in its place, not its text
        self.assertEqual(e.last["finish"], "stop")
        self.assertEqual(run(e, [1]), [THINK, HELLO, IM_END])   # a special token that is not an end comes in order

    def test_a_consumer_that_leaves_ends_the_request(self):
        fake = Fake(self, [[HELLO] * 400], "--fake-delay-s", "0.01")
        e = fake.engine(self)
        gen = e.generate([1, 2], 400, {}, threading.Event())
        got = [t for _, t in zip(range(6), filter(None, gen))]
        gen.close()
        self.assertEqual(got, [HELLO] * 6)
        self.assertEqual(e.last["finish"], "stop")
        self.assertEqual(e.last["generated"], 6)
        S.time.sleep(0.5)
        self.assertEqual(run(e, [1], max_new=3), [HELLO] * 3)             # the engine goes on with the next one
        first = fake.requests()[0]
        self.assertEqual(first["end"], "client left")
        self.assertLess(first["generated"], 400)

    def test_cancel(self):
        fake = Fake(self, [[HELLO] * 400], "--fake-delay-s", "0.01")
        e = fake.engine(self)
        cancel = threading.Event()
        out = []
        for t in e.generate([1], 400, {}, cancel):
            if t is not None:
                out.append(t)
                if len(out) == 3:
                    cancel.set()
        self.assertEqual(len(out), 3)
        self.assertEqual(e.last["finish"], "cancel")

    def test_a_refused_request_is_the_engines_error(self):
        fake = Fake(self, [[HELLO, IM_END]], ctx=16)
        e = fake.engine(self)
        with self.assertRaisesRegex(ValueError, "exceeds the available context size"):
            run(e, list(range(40)))
        self.assertTrue(e.alive())

    def test_llama_server_that_dies_mid_reply(self):
        fake = Fake(self, [[HELLO] * 10], "--fake-die-after", "3")
        e = fake.engine(self)
        with self.assertRaises(S.EngineDied):
            run(e, [1])
        self.assertFalse(e.alive())
        self.assertIn("exit code 3", e.death_note())


class ThroughTheService(unittest.TestCase):
    """The whole front end over llama-server: the template, the stop tokens, the parser."""

    def service(self, replies):
        fake = Fake(self, replies)
        tok = tokenizer()
        e = fake.engine(self)
        svc = S.Service(e, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        return svc, tok, fake

    def test_a_vocabulary_without_endoftext_stops_only_at_im_end(self):
        # Nemotron 3.5 has <|im_end|> and no <|endoftext|>: the plain tokens of that text are not stop tokens
        svc, tok, _ = self.service([[ord("<"), ord("|"), HELLO, IM_END]])
        self.assertEqual(svc.stop_ids, {IM_END})
        self.assertEqual(svc.engine.eos_ids, frozenset({IM_END}))
        events = list(svc.run(tok.encode("hi"), False, None, 32, {}, threading.Event()))
        text = "".join(ev.text or "" for kind, ev in events if kind == "event" and ev.kind == "content")
        done = next(x for kind, x in events if kind == "done")
        self.assertEqual(text, "<|Hello")
        self.assertEqual((done["finish"], done["completion_tokens"]), ("stop", 4))
        self.assertEqual(done["timings"]["prompt_ms"], 12.5)


class Registry(unittest.TestCase):
    def test_a_llama_server_model_is_listed_with_its_file_and_context(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            model = gguf(root / "bolt.gguf", general__architecture="nemotron_h_moe", general__name="Bolt",
                         general__size_label="74B-A9B")
            (root / "tok").mkdir()
            (root / "tok" / "vocab.json").write_text("{}")
            (root / "llama-server").write_text("")
            cfg = root / "strata-bolt.json"
            cfg.write_text(json.dumps({"engine": "llama-server", "exe": str(root / "llama-server"),
                                       "args": ["-m", str(model), "-c", "168000", "-np", "1"],
                                       "tokenizer": str(root / "tok"), "model_name": "bolt"}))
            e = registry.describe(cfg)
            self.assertEqual((e["status"], e["backend"], e["path"], e["context"]),
                             ("available", "nemotron_h_moe", str(model), 168000))
            # the same file under strata-poly, which runs nemotron_h_moe too (Bolt's default engine since 2026-11-21)
            other = root / "strata-other.json"
            other.write_text(json.dumps({"exe": str(root / "llama-server"), "args": ["--model", str(model)],
                                         "tokenizer": str(root / "tok")}))
            o = registry.describe(other)
            self.assertEqual((o["status"], o["backend"], o["path"]), ("available", "nemotron_h_moe", str(model)))


if __name__ == "__main__":
    unittest.main()
