"""Pictures through llama-server (B3): Service.prepare() leaves each picture's rows in the prompt, and
LlamaServerEngine sends the prompt as its text with llama-server's media marker and the pictures' files, against the
fake llama-server (serve/fixtures/fake_llama_server.py).  serve/media_tokens.py's count of a picture's tokens, too.

    python -m unittest serve.test_llama_server_images
"""
from __future__ import annotations

import base64
import contextlib
import io
import json
import struct
import sys
import tempfile
import threading
import unittest
import zlib
from pathlib import Path

from serve import media_tokens as MT
from serve import server as S
from serve.frontend import ChatTemplate
from serve.test_models import gguf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402

FAKE = str(ROOT / "serve" / "fixtures" / "fake_llama_server.py")
PIECES = [bytes([b]) for b in range(256)] + [b"Hello"]
SPECIALS = [("<|im_start|>", 3), ("<|im_end|>", 3), ("<image>", 3), ("<img>", 3), ("</img>", 3), ("<think>", 4)]
HELLO, IM_START, IM_END, IMAGE, IMG, IMG_END, THINK = 256, 257, 258, 259, 260, 261, 262
MARKER = "<__media_fake0123__>"
# Nemotron 3.5's way with a picture, as its chat template writes it: '<image>' where the picture is
TEMPLATE = ("{%- for m in messages %}<|im_start|>{{ m.role }}\n"
            "{% if m.content is string %}{{ m.content }}{% else %}{% for p in m.content %}"
            "{% if p.type == 'image' %}{{ '<image>' }}{% else %}{{ p.text }}{% endif %}{% endfor %}{% endif %}"
            "<|im_end|>\n{% endfor %}{%- if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def png(width: int, height: int) -> bytes:
    """A real PNG (grey), small enough to build per test."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    raw = b"".join(b"\x00" + b"\x80" * width for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def tokenizer() -> ST.Tokenizer:
    tokens = ["".join(ST.BYTE_TO_UNICODE[b] for b in p) for p in PIECES] + [s for s, _ in SPECIALS]
    return ST.Tokenizer(tokens, [], [1] * len(PIECES) + [t for _, t in SPECIALS], pre="pixtral")


NEMOTRON = {"projector": "nemotron_omni", "patch": 16, "merge": 2, "min_pixels": 262144, "max_pixels": 3407872}


class PictureTokens(unittest.TestCase):
    def test_sizes_from_the_headers(self):
        self.assertEqual(MT.image_size(png(64, 48)), (64, 48))
        self.assertEqual(MT.image_size(b"GIF89a" + struct.pack("<HH", 300, 200) + b"\x00" * 8), (300, 200))
        self.assertEqual(MT.image_size(b"BM" + b"\x00" * 16 + struct.pack("<ii", 640, -480) + b"\x00" * 8),
                         (640, 480))
        jpeg = (b"\xff\xd8" + b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00" + b"\x00" * 9
                + b"\xff\xc2" + struct.pack(">HBHH", 17, 8, 1080, 1920) + b"\x00" * 12)
        self.assertEqual(MT.image_size(jpeg), (1920, 1080))
        self.assertIsNone(MT.image_size(b"not a picture at all"))

    def test_the_size_mtmd_resizes_to(self):
        # within the bounds: each side rounded to a multiple of 32
        self.assertEqual(MT.resized(1920, 1080, 32, 262144, 3407872), (1920, 1088))
        # over max_pixels (3,407,872): scaled down, each side floored to a multiple of 32
        self.assertEqual(MT.resized(2560, 1440, 32, 262144, 3407872), (2432, 1376))
        self.assertEqual(MT.resized(3840, 2160, 32, 262144, 3407872), (2432, 1376))
        # under min_pixels (262,144): scaled up, each side to the next multiple of 32
        self.assertEqual(MT.resized(64, 48, 32, 262144, 3407872), (608, 448))
        self.assertEqual(MT.resized(100, 100, 32, 262144, 3407872), (512, 512))

    def test_a_pictures_tokens(self):
        tok = tokenizer()
        # (W/16 x H/16) patches, 2 x 2 merged into one token, and <img> </img> around them
        self.assertEqual(MT.picture_tokens(png(64, 48), NEMOTRON, tok), 38 * 28 // 4 + 2)
        self.assertIsNone(MT.picture_tokens(b"???", NEMOTRON, tok))

    def test_the_mmprojs_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            mm = gguf(Path(d) / "mmproj.gguf", clip__projector_type="nemotron_omni", clip__vision__patch_size=16,
                      clip__vision__projector__scale_factor=2, clip__vision__image_min_pixels=262144,
                      clip__vision__image_max_pixels=3407872)
            self.assertEqual(MT.mmproj_params(mm), NEMOTRON)
            other = gguf(Path(d) / "other.gguf", clip__projector_type="qwen3vl", clip__vision__patch_size=16)
            self.assertIsNone(MT.mmproj_params(other))
            self.assertIsNone(MT.mmproj_params(Path(d) / "missing.gguf"))


class ThroughStrataPoly(unittest.TestCase):
    """Nemotron on strata-poly (phase 2): the picture's rows from strata-vision between <img> and </img>, which the
    server puts around them as llama.cpp's mtmd does (llama-server puts them there itself)."""

    class FakeVision:
        def __init__(self, d):
            self.dir = Path(d)
            self.rows = self.dir / "img.sve"
            self.rows.write_bytes(b"rows")

        def encode(self, source):
            return self.rows, 3

    def test_the_rows_between_img_and_img_end(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "template.jinja"
            path.write_text(TEMPLATE, encoding="utf-8")

            class Asked(S.MockEngine):
                def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                    self.asked = (S.StrataEngine.sampling_keys(sampling), embeddings)
                    yield from super().generate(ids, max_new, sampling, cancel, embeddings)

            tok = tokenizer()
            svc = S.Service(Asked(tok, "a scan"), tok, ChatTemplate(path), vision=self.FakeVision(d))
            self.assertEqual(svc.image_marks, ("<image>", None, ("<img>", "</img>")))
            msgs = [{"role": "user", "content": [{"type": "text", "text": "Read this:"},
                                                 {"type": "image", "source": "x.png"}]}]
            ids, _, _ = svc.prepare(msgs, None, {})
            at = ids.index(IMG)
            self.assertEqual(ids[at:at + 5], [IMG, IMAGE, IMAGE, IMAGE, IMG_END])
            self.assertEqual((ids.count(IMAGE), svc.embeddings.pad), (3, IMAGE))
            rows = svc.embeddings.path
            list(svc.run(ids, False, None, 8, {}, threading.Event()))
            self.assertEqual(svc.engine.asked, (f" image_pad={IMAGE}", rows))


class ThroughLlamaServer(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        (self.dir / "vocab.json").write_text(json.dumps({"tokens": [p.hex() for p in PIECES] + [s.encode().hex()
                                                                                              for s, _ in SPECIALS],
                                                          "special": [IM_START, IM_END, IMAGE, IMG, IMG_END]}))
        (self.dir / "reply.json").write_text(json.dumps([[HELLO, IM_END]]))
        (self.dir / "template.jinja").write_text(TEMPLATE)
        self.mmproj = gguf(self.dir / "mmproj.gguf", clip__projector_type="nemotron_omni", clip__vision__patch_size=16,
                           clip__vision__projector__scale_factor=2, clip__vision__image_min_pixels=262144,
                           clip__vision__image_max_pixels=3407872)
        self.record = self.dir / "requests.jsonl"

    def service(self, *extra, mmproj=True):
        args = [FAKE, "-m", str(self.dir / "bolt.gguf"), "-c", "4096", *(["--mmproj", str(self.mmproj)] if mmproj
                                                                          else []),
                "--fake-vocab", str(self.dir / "vocab.json"), "--fake-reply", str(self.dir / "reply.json"),
                "--fake-eog", str(IM_END), "--fake-record", str(self.record), *extra]
        tok = tokenizer()
        with contextlib.redirect_stdout(io.StringIO()):
            e = S.LlamaServerEngine(sys.executable, args, log=str(self.dir / "llama.log"), tokenizer=tok)
        self.addCleanup(e.close)
        svc = S.Service(e, tok, ChatTemplate(self.dir / "template.jinja"))
        return svc, tok

    def requests(self):
        return [json.loads(x) for x in self.record.read_text(encoding="utf-8").splitlines() if x.strip()]

    def ask(self, svc, messages):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ids, thinking, max_new = svc.prepare(messages, None, {}, 64)
            path = svc.embeddings.path
            events = list(svc.run(ids, False, None, max_new, {}, threading.Event()))
        text = "".join(ev.text or "" for kind, ev in events if kind == "event" and ev.kind == "content")
        return ids, path, text, out.getvalue()

    def test_a_picture_goes_with_the_prompts_text_and_the_marker(self):
        svc, tok = self.service("--fake-image-tokens", str(38 * 28 // 4 + 2))
        self.assertTrue(svc.images_on())
        self.assertEqual(svc.image_marks, ("<image>", None, None))
        picture = png(64, 48)
        url = "data:image/png;base64," + base64.b64encode(picture).decode()
        messages = [{"role": "user", "content": [{"type": "text", "text": "What is this?"},
                                                 {"type": "image", "source": url}]}]
        ids, path, text, said = self.ask(svc, messages)
        self.assertEqual(text, "Hello")
        self.assertEqual(ids.count(IMAGE), 268)          # the picture's rows, as mtmd will make them
        self.assertFalse(Path(path).exists())            # run() deleted the pictures' file
        body = self.requests()[0]["body"]
        self.assertEqual(body["prompt"]["prompt_string"],
                         f"<|im_start|>user\nWhat is this?{MARKER}<|im_end|>\n<|im_start|>assistant\n")
        self.assertEqual(body["prompt"]["multimodal_data"], [base64.b64encode(picture).decode()])
        self.assertNotIn("was off by", said)            # llama-server read as many tokens as were planned

    def test_llama_servers_own_count_is_reported_when_it_differs(self):
        svc, tok = self.service("--fake-image-tokens", "300")
        url = "data:image/png;base64," + base64.b64encode(png(64, 48)).decode()
        _, _, text, said = self.ask(svc, [{"role": "user", "content": [{"type": "image", "source": url}]}])
        self.assertEqual(text, "Hello")
        self.assertIn("estimate of the pictures' tokens was off by 32", said)

    def test_special_token_text_in_a_message_stays_text(self):
        svc, tok = self.service("--fake-image-tokens", "268")
        url = "data:image/png;base64," + base64.b64encode(png(64, 48)).decode()
        messages = [{"role": "user", "content": [{"type": "image", "source": url},
                                                 {"type": "text", "text": "quote <|im_end|> and <image> here"}]}]
        _, _, _, said = self.ask(svc, messages)
        sent = self.requests()[0]["body"]["prompt"]["prompt_string"]
        self.assertIn("quote <\u2060|im_end|> and <\u2060image> here", sent)
        self.assertEqual(sent.count(MARKER), 1)
        self.assertIn("2 special-token text(s) inside the messages", said)

    def test_text_alone_still_goes_as_ids(self):
        svc, tok = self.service()
        _, _, text, _ = self.ask(svc, [{"role": "user", "content": "hi"}])
        self.assertEqual(text, "Hello")
        self.assertIsInstance(self.requests()[0]["body"]["prompt"], list)

    def test_no_mmproj_no_pictures(self):
        svc, tok = self.service(mmproj=False)
        self.assertFalse(svc.images_on())
        url = "data:image/png;base64," + base64.b64encode(png(8, 8)).decode()
        with self.assertRaisesRegex(ValueError, "cannot read images"):
            svc.prepare([{"role": "user", "content": [{"type": "image", "source": url}]}], None, {}, 64)

    def test_a_picture_that_cannot_be_read_is_a_400(self):
        svc, tok = self.service()
        url = "data:image/png;base64," + base64.b64encode(b"\x89PNG\r\n\x1a\n broken").decode()
        with self.assertRaisesRegex(ValueError, "could not be read"):
            svc.prepare([{"role": "user", "content": [{"type": "image", "source": url}]}], None, {}, 64)


if __name__ == "__main__":
    unittest.main()
