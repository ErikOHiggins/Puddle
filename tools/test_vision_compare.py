"""Tests for tools/vision_compare.py, against a stand-in for strata-vision (its READY / ENC / OK / QUIT protocol and its
records): two encoders that agree, and two that do not.

    python -m unittest tools.test_vision_compare
"""
from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vision_compare as V  # noqa: E402

FAKE = '''import struct, sys
scale = float(sys.argv[sys.argv.index("--scale") + 1]) if "--scale" in sys.argv else 1.0
print("READY 4", flush=True)
for line in sys.stdin:
    line = line.strip()
    if line == "QUIT":
        break
    _, img, out = line.split(" ")
    rows = [0.5 * scale, -1.25, 3.0, 0.0] * 3
    with open(out, "wb") as f:
        f.write(struct.pack("<5i", 0x31455653, 3, 3, 1, 4) + struct.pack("<12f", *rows))
    print("OK 3 3 1 7", flush=True)
'''


class Records(unittest.TestCase):
    def test_read_and_compare(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.sve"
            p.write_bytes(struct.pack("<5i", 0x31455653, 2, 2, 1, 2) + struct.pack("<4f", 1.0, 2.0, 3.0, 4.0))
            (n, nx, ny, rows), = V.read_records(p)
            self.assertEqual((n, nx, ny, list(rows)), (2, 2, 1, [1.0, 2.0, 3.0, 4.0]))
        c = V.compare_rows(rows, rows)
        self.assertEqual((c["max_abs"], c["same_share"]), (0.0, 1.0))
        other = type(rows)("f", [1.0, 2.5, 3.0, 4.0])
        c = V.compare_rows(rows, other)
        self.assertEqual((c["max_abs"], c["same_share"]), (0.5, 0.75))

    def test_the_encoders_command(self):
        v = {"mmproj": "m.gguf", "model": "g.gguf", "gpu": True, "max_tokens": 1024}
        self.assertEqual(V.encoder_cmd("enc", v), ["enc", "--mmproj", "m.gguf", "--model", "g.gguf", "--gpu",
                                                   "--max-tokens", "1024"])
        self.assertNotIn("--max-tokens", V.encoder_cmd("enc", {**v, "max_tokens": 0}))


class AgainstAStandIn(unittest.TestCase):
    def run_it(self, old_args, new_args):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            fake = d / "fake_vision.py"
            fake.write_text(FAKE, encoding="utf-8")
            cfg = d / "strata-x.json"
            vision = {"exe": "unused", "mmproj": "m", "model": "g"}
            cfg.write_text(json.dumps({"exe": "x", "args": [], "vision": vision}))
            img = d / "pic.png"
            img.write_bytes(b"png")
            # an encoder is a command line: the stand-in run by this Python, with the given extra arguments
            orig = V.encoder_cmd
            V.encoder_cmd = lambda exe, vision: [sys.executable, str(fake), *orig("x", vision)[1:],
                                                *(old_args if exe == "old" else new_args)]
            try:
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    rc = V.main([str(cfg), "--old", "old", "--new", "new", "--image", str(img)])
            finally:
                V.encoder_cmd = orig
            return rc, out.getvalue()

    def test_the_same_rows(self):
        rc, out = self.run_it([], [])
        self.assertEqual(rc, 0, out)
        self.assertIn("the same to the bit: 100.00%", out)

    def test_other_rows(self):
        rc, out = self.run_it([], ["--scale", "2"])
        self.assertEqual(rc, 1, out)
        self.assertIn("the same to the bit: 75.00%", out)
        self.assertIn("largest difference 0.5", out)


if __name__ == "__main__":
    unittest.main()
