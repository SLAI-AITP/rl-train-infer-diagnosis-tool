"""Behavioral checks for the numerical helper; no engine integration claims."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from compare_tensors import compare, load_snapshot


def snapshot(values, mask=None, keys=None):
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    return {
        "values": values,
        "token_keys": np.asarray(keys if keys is not None else [f"s:{i}" for i in range(n)]),
        "valid_mask": np.asarray(mask if mask is not None else [True] * n, dtype=bool),
    }


class ComparisonTests(unittest.TestCase):
    def test_equal_token_weight_not_flattened(self):
        a = snapshot([[1000, 0], [1, 0]])
        b = snapshot([[1000, 0], [0, 1]])
        report, _ = compare(a, b)
        self.assertAlmostEqual(report["cosine_mean"], 0.5)

    def test_chunking_preserves_token_weights(self):
        a = snapshot([[1, 0]] * 5)
        b = snapshot([[1, 0], [1, 0], [0, 1], [-1, 0], [0, 1]])
        r1, _ = compare(a, b, chunk_rows=1)
        r2, _ = compare(a, b, chunk_rows=3)
        self.assertEqual(r1, r2)
        self.assertAlmostEqual(r1["cosine_mean"], 0.2)

    def test_padding_nan_is_excluded(self):
        a = snapshot([[1, 2], [np.nan, np.inf]], mask=[True, False])
        r, p = compare(a, a)
        self.assertEqual(r["status"], "ok")
        self.assertEqual(r["n_valid"], 1)
        self.assertAlmostEqual(r["cosine_mean"], 1.0)
        self.assertTrue(np.isnan(p["cosine"][1]))

    def test_mask_difference_rejected(self):
        with self.assertRaisesRegex(ValueError, "intersection"):
            compare(snapshot([[1], [2]]), snapshot([[1], [2]], mask=[True, False]))

    def test_reordered_keys_rejected(self):
        with self.assertRaisesRegex(ValueError, "reordered"):
            compare(snapshot([[1], [2]]), snapshot([[1], [2]], keys=["s:1", "s:0"]))

    def test_duplicate_keys_rejected(self):
        a = snapshot([[1], [2]], keys=["same", "same"])
        with self.assertRaisesRegex(ValueError, "unique"):
            compare(a, a)

    def test_shape_broadcasting_rejected(self):
        with self.assertRaisesRegex(ValueError, "broadcasting"):
            compare(snapshot([[1, 2]]), snapshot([[1]]))

    def test_no_valid_tokens_rejected(self):
        a = snapshot([[1]], mask=[False])
        with self.assertRaisesRegex(ValueError, "no valid tokens"):
            compare(a, a)

    def test_zero_norm_does_not_become_agreement(self):
        a = snapshot([[0, 0], [1, 0]])
        r, _ = compare(a, a, auxiliary=True)
        self.assertIsNone(r["cosine_mean"])
        self.assertEqual(r["cosine_defined_subset_mean"], 1.0)
        self.assertEqual(r["n_zero_norm_finite_pairs"], 1)
        self.assertEqual(r["auxiliary"]["max_abs_error"], 0.0)
        self.assertEqual(r["auxiliary"]["n_undefined_relative_l2"], 1)
        json.dumps(r, allow_nan=False)

    def test_valid_nonfinite_not_silently_dropped(self):
        a = snapshot([[np.nan, 0], [1, 0]])
        r, _ = compare(a, a, auxiliary=True)
        self.assertEqual(r["status"], "undefined_cosine")
        self.assertIsNone(r["cosine_mean"])
        self.assertEqual(r["n_nonfinite_valid_pairs"], 1)
        self.assertIsNone(r["auxiliary"]["max_abs_error"])
        json.dumps(r, allow_nan=False)

    def test_scale_difference_requires_auxiliary(self):
        a, b = snapshot([[1, 2], [3, 4]]), snapshot([[2, 4], [6, 8]])
        r, _ = compare(a, b)
        self.assertAlmostEqual(r["cosine_mean"], 1.0)
        self.assertNotIn("auxiliary", r)
        r, _ = compare(a, b, auxiliary=True)
        self.assertAlmostEqual(r["auxiliary"]["relative_l2_train_denominator"]["mean"], 1.0)
        self.assertAlmostEqual(r["auxiliary"]["infer_over_train_norm"]["mean"], 2.0)

    def test_float16_cast_before_difference(self):
        a, b = snapshot([[60000, 1]]), snapshot([[-60000, 1]])
        a["values"] = a["values"].astype(np.float16)
        b["values"] = b["values"].astype(np.float16)
        r, _ = compare(a, b, auxiliary=True)
        self.assertEqual(r["auxiliary"]["max_abs_error"], 120000.0)

    def test_auxiliary_rows_keep_original_positions_and_undefined_values(self):
        mask = [True, False, True, True, True]
        a = snapshot([[1, 2], [np.nan, 0], [0, 0], [np.inf, 1], [3, 4]], mask=mask)
        b = snapshot([[2, 4], [np.nan, 0], [1, 0], [1, 1], [3, 4]], mask=mask)
        r, p = compare(a, b, auxiliary=True, chunk_rows=1)
        np.testing.assert_allclose(p["relative_l2"], [1, np.nan, np.nan, np.nan, 0], equal_nan=True)
        np.testing.assert_allclose(p["norm_ratio"], [2, np.nan, np.nan, np.nan, 1], equal_nan=True)
        np.testing.assert_allclose(p["max_abs_error"], [2, np.nan, 1, np.nan, 0], equal_nan=True)
        self.assertEqual(r["auxiliary"]["n_undefined_relative_l2"], 2)
        self.assertEqual(r["auxiliary"]["n_undefined_norm_ratio"], 2)
        self.assertIsNone(r["auxiliary"]["relative_l2_train_denominator"])
        r2, p2 = compare(a, b, auxiliary=True, chunk_rows=3)
        self.assertEqual(r, r2)
        for key in ("cosine", "relative_l2", "norm_ratio", "max_abs_error"):
            np.testing.assert_allclose(p[key], p2[key], equal_nan=True)

    def test_auxiliary_aggregate_excludes_padding(self):
        a = snapshot([[1, 2], [np.nan, np.inf], [3, 4]], mask=[True, False, True])
        b = snapshot([[2, 4], [np.nan, np.inf], [3, 4]], mask=[True, False, True])
        r, p = compare(a, b, auxiliary=True)
        self.assertEqual(r["auxiliary"]["relative_l2_train_denominator"]["mean"], 0.5)
        self.assertEqual(r["auxiliary"]["infer_over_train_norm"]["mean"], 1.5)
        self.assertEqual(r["auxiliary"]["n_undefined_relative_l2"], 0)
        self.assertTrue(np.isnan(p["max_abs_error"][1]))

    def test_cosine_handles_large_finite_values(self):
        a = snapshot([[1e300, 1e300]])
        r, _ = compare(a, a)
        self.assertAlmostEqual(r["cosine_mean"], 1.0)

    def test_logprob_sign_and_mask(self):
        a = snapshot([-1, -3, np.nan], mask=[True, True, False])
        b = snapshot([-2, -1, np.inf], mask=[True, True, False])
        r, p = compare(a, b, kind="logprobs")
        self.assertEqual(r["delta_logprob_mean"], -0.5)
        self.assertEqual(r["abs_delta_logprob"]["mean"], 1.5)
        np.testing.assert_array_equal(p["delta_logprob"][:2], [1, -2])

    def test_logprob_underflow_preserves_log_difference(self):
        r, _ = compare(snapshot([-1000]), snapshot([-1001]), kind="logprobs")
        self.assertEqual(r["abs_delta_logprob"]["mean"], 1.0)
        self.assertEqual(r["abs_delta_p"]["mean"], 0.0)

    def test_nonfinite_logprob_invalidates_aggregate(self):
        r, _ = compare(snapshot([-np.inf, -1]), snapshot([-2, -1]), kind="logprobs")
        self.assertEqual(r["status"], "nonfinite_logprob_comparison")
        self.assertIsNone(r["abs_delta_logprob"])
        json.dumps(r, allow_nan=False)

    def test_positive_logprob_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive logprob"):
            compare(snapshot([0.1]), snapshot([-1]), kind="logprobs")

    def test_cli_npz_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            a, b = snapshot([[1, 0], [1, 0]]), snapshot([[1, 0], [0, 1]])
            np.savez(base / "train.npz", **a)
            np.savez(base / "infer.npz", **b)
            np.testing.assert_array_equal(load_snapshot(base / "train.npz")["values"], a["values"])
            proc = subprocess.run([
                sys.executable, str(Path(__file__).with_name("compare_tensors.py")),
                "--kind", "hidden", "--train", str(base / "train.npz"),
                "--infer", str(base / "infer.npz"), "--output", str(base / "metrics.json"),
                "--per-token", str(base / "tokens.npz"),
            ], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            r = json.loads((base / "metrics.json").read_text())
            self.assertEqual(r["cosine_mean"], 0.5)
            with np.load(base / "tokens.npz", allow_pickle=False) as output:
                np.testing.assert_array_equal(output["cosine"], [1, 0])

    def test_cli_auxiliary_export_and_path_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            np.savez(base / "train.npz", **snapshot([[1, 2], [3, 4]]))
            np.savez(base / "infer.npz", **snapshot([[2, 4], [3, 4]]))
            command = [
                sys.executable, str(Path(__file__).with_name("compare_tensors.py")),
                "--kind", "hidden", "--train", str(base / "train.npz"),
                "--infer", str(base / "infer.npz"), "--output", str(base / "metrics.json"),
                "--auxiliary", "--per-token", str(base / "tokens.npz"),
            ]
            proc = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            r = json.loads(proc.stdout)
            self.assertEqual(r["schema_version"], 2)
            self.assertEqual(r["path_mode"], "omit")
            self.assertNotIn("train_path", r)
            self.assertNotIn("infer_path", r)
            self.assertNotIn(str(base), proc.stdout)
            with np.load(base / "tokens.npz", allow_pickle=False) as output:
                np.testing.assert_array_equal(output["relative_l2"], [1, 0])
                np.testing.assert_array_equal(output["norm_ratio"], [2, 1])
                np.testing.assert_array_equal(output["max_abs_error"], [2, 0])
                np.testing.assert_array_equal(output["token_keys"], ["s:0", "s:1"])
            proc = subprocess.run(command + ["--path-mode", "absolute"], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            r = json.loads(proc.stdout)
            self.assertEqual(r["train_path"], str((base / "train.npz").resolve()))
            self.assertEqual(r["infer_path"], str((base / "infer.npz").resolve()))


if __name__ == "__main__":
    unittest.main()
