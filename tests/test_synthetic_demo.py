import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class SyntheticDemoTests(unittest.TestCase):
    def test_saved_evidence_supports_source_amplification_and_reversal(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "demo"
            command = [sys.executable, str(ROOT / "examples/synthetic/run_demo.py"),
                       "--output", str(out)]
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["data_origin"], "synthetic")
            self.assertFalse(summary["engine_integration_tested"])
            self.assertGreater(summary["amplifier_G"], summary["source_G"])
            replay = json.loads((out / "replay.json").read_text())
            for key in ("R_T", "R_I", "S_T", "S_I"):
                self.assertAlmostEqual(replay[key]["cosine_mean"], 1)
                self.assertEqual(replay[key]["auxiliary"]["max_abs_error"], 0)
            for key in ("P_T", "P_I"):
                self.assertLess(replay[key]["cosine_mean"], replay["C_in"]["cosine_mean"])
            endpoint = summary["endpoint"]
            self.assertGreater(endpoint["baseline"], 0)
            self.assertEqual(endpoint["intervention"], 0)
            self.assertEqual(endpoint["revert"], endpoint["baseline"])
            self.assertGreater(endpoint["extra_baseline"], 0)
            self.assertEqual(endpoint["extra_intervention"], 0)
            with np.load(out / "scale/per_token.npz", allow_pickle=False) as data:
                np.testing.assert_array_equal(data["relative_l2"], [1, 0])
                np.testing.assert_array_equal(data["norm_ratio"], [2, 1])
                np.testing.assert_array_equal(data["max_abs_error"], [2, 0])
            # Independently verify the endpoint against the saved target logprobs.
            with np.load(out / "chain/train_logprobs.npz", allow_pickle=False) as train:
                with np.load(out / "chain/infer_logprobs.npz", allow_pickle=False) as infer:
                    self.assertAlmostEqual(np.mean(np.abs(train["values"] - infer["values"])),
                                           endpoint["baseline"])
            original = (out / "summary.json").read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertEqual((out / "summary.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
