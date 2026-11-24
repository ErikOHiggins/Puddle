"""#465: setup's recommendation for "parallel" (several requests decoded together in the engine's batch slots).
No GPU or download needed: python tools/test_setup_parallel.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402


class Parallel(unittest.TestCase):


    def test_poly_notes(self):
        d = S.family_models("occamy")["APEX-Quality"]
        # 8 GB: most experts on the CPU, so requests at once gain less
        self.assertIn("gain less", S.poly_parallel_note(None, d, 8.0)[0])
        self.assertTrue(any("gain less" in x for x in S.poly_parallel_note(3, d, 8.0)))
        # 32 GB holds every expert: worth it, and asked for, nothing to warn of
        self.assertIn("--parallel 2 to 4", S.poly_parallel_note(None, d, 31.8)[0])
        self.assertEqual(len(S.poly_parallel_note(3, d, 31.8)), 1)
        self.assertTrue(any("at most 8" in x for x in S.poly_parallel_note(12, d, 31.8)))


if __name__ == "__main__":
    unittest.main()
