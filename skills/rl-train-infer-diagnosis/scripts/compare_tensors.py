#!/usr/bin/env python3
"""Compare semantically aligned NPZ snapshots. Requires Python 3 and NumPy.

Each input contains floating values, Unicode token_keys, and a boolean valid_mask.
This tool measures differences; it does not decide equivalence or causality.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load_snapshot(path):
    with np.load(path, allow_pickle=False) as data:
        required = {"values", "token_keys", "valid_mask"}
        if not required.issubset(data.files):
            raise ValueError(f"{path}: required arrays are {sorted(required)}")
        return {key: data[key] for key in required}


def validate_pair(train, infer, kind):
    if kind not in {"hidden", "logprobs"}:
        raise ValueError("kind must be hidden or logprobs")
    ndim = 2 if kind == "hidden" else 1
    for label, item in (("train", train), ("infer", infer)):
        values, keys, mask = (item[k] for k in ("values", "token_keys", "valid_mask"))
        if values.dtype.kind != "f" or values.ndim != ndim:
            raise ValueError(f"{label}: values must be floating point with ndim={ndim}")
        if kind == "hidden" and values.shape[1] == 0:
            raise ValueError(f"{label}: hidden feature dimension must be nonempty")
        if keys.shape != (values.shape[0],) or keys.dtype.kind != "U":
            raise ValueError(f"{label}: token_keys must be a Unicode [N] array")
        if np.any(keys == "") or np.unique(keys).size != keys.size:
            raise ValueError(f"{label}: token_keys must be nonempty and unique")
        if mask.shape != keys.shape or mask.dtype.kind != "b":
            raise ValueError(f"{label}: valid_mask must be boolean [N]")
    if train["values"].shape != infer["values"].shape:
        raise ValueError("values shapes differ; broadcasting is not permitted")
    if not np.array_equal(train["token_keys"], infer["token_keys"]):
        raise ValueError("token_keys differ or are reordered; align semantically before comparing")
    if not np.array_equal(train["valid_mask"], infer["valid_mask"]):
        raise ValueError("valid_mask differs; automatic intersection is not permitted")
    if not np.any(train["valid_mask"]):
        raise ValueError("no valid tokens")


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def stats(values):
    """Do not silently remove undefined or nonfinite observations."""
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        return {
            "mean": finite_number(np.mean(values, dtype=np.float64)),
            "p95": finite_number(np.percentile(values, 95)),
            "p99": finite_number(np.percentile(values, 99)),
            "max": finite_number(np.max(values)),
        }


def stable_norm(values):
    scale = np.max(np.abs(values), axis=1)
    safe_scale = np.where(scale > 0, scale, 1.0)
    with np.errstate(over="ignore", invalid="ignore"):
        return scale * np.sqrt(np.sum((values / safe_scale[:, None]) ** 2, axis=1))


def compare_hidden(train, infer, auxiliary, chunk_rows):
    mask = train["valid_mask"]
    indices = np.flatnonzero(mask)
    cosine = np.full(mask.shape, np.nan, dtype=np.float64)
    nonfinite_count = zero_count = 0
    rel_l2 = np.full(mask.shape, np.nan) if auxiliary else None
    norm_ratio = np.full(mask.shape, np.nan) if auxiliary else None
    row_max_abs = np.full(mask.shape, np.nan) if auxiliary else None
    max_abs = 0.0
    for start in range(0, indices.size, chunk_rows):
        row_ids = indices[start:start + chunk_rows]
        a = train["values"][row_ids].astype(np.float64)
        b = infer["values"][row_ids].astype(np.float64)
        finite = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
        nonfinite_count += int(np.sum(~finite))
        finite_pos = np.flatnonzero(finite)
        af, bf = a[finite], b[finite]
        if not finite_pos.size:
            continue
        sa = np.max(np.abs(af), axis=1)
        sb = np.max(np.abs(bf), axis=1)
        defined = (sa > 0) & (sb > 0)
        zero_count += int(np.sum(~defined))
        if np.any(defined):
            # Scale before normalization: avoid overflow/underflow in dot and norms.
            ua = af[defined] / sa[defined, None]
            ub = bf[defined] / sb[defined, None]
            ua /= np.sqrt(np.sum(ua * ua, axis=1))[:, None]
            ub /= np.sqrt(np.sum(ub * ub, axis=1))[:, None]
            result = np.clip(np.sum(ua * ub, axis=1), -1.0, 1.0)
            cosine[row_ids[finite_pos[defined]]] = result
        if auxiliary:
            na, nb = stable_norm(af), stable_norm(bf)
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                delta = bf - af  # Cast to float64 before subtraction.
                err = stable_norm(delta)
                rel = np.where(na > 0, err / na, np.nan)
                ratio = np.where(na > 0, nb / na, np.nan)
                row_errors = np.max(np.abs(delta), axis=1)
                max_abs = max(max_abs, float(np.max(row_errors)))
            finite_ids = row_ids[finite_pos]
            rel_l2[finite_ids] = rel
            norm_ratio[finite_ids] = ratio
            row_max_abs[finite_ids] = row_errors
    selected = cosine[mask]
    defined_values = selected[np.isfinite(selected)]
    complete = defined_values.size == indices.size
    mean = float(np.mean(defined_values)) if defined_values.size else None
    report = {
        "status": "ok" if complete else "undefined_cosine",
        "cosine_mean": mean if complete else None,
        "one_minus_cosine_mean": (1.0 - mean) if complete else None,
        "cosine_defined_subset_mean": mean,
        "n_cosine_defined": int(defined_values.size),
        "n_nonfinite_valid_pairs": nonfinite_count,
        "n_zero_norm_finite_pairs": zero_count,
        "cosine_min_defined": float(np.min(defined_values)) if defined_values.size else None,
    }
    per_token = {"cosine": cosine}
    if auxiliary:
        report["auxiliary"] = {
            "relative_l2_train_denominator": stats(rel_l2[mask]),
            "infer_over_train_norm": stats(norm_ratio[mask]),
            "n_undefined_relative_l2": int(np.sum(~np.isfinite(rel_l2[mask]))),
            "n_undefined_norm_ratio": int(np.sum(~np.isfinite(norm_ratio[mask]))),
            "max_abs_error": finite_number(max_abs) if not nonfinite_count else None,
        }
        per_token.update({
            "relative_l2": rel_l2,
            "norm_ratio": norm_ratio,
            "max_abs_error": row_max_abs,
        })
    return report, per_token


def compare_logprobs(train, infer):
    mask = train["valid_mask"]
    a = train["values"][mask].astype(np.float64)
    b = infer["values"][mask].astype(np.float64)
    if np.any(a[np.isfinite(a)] > 0) or np.any(b[np.isfinite(b)] > 0):
        raise ValueError("positive logprob found; check that inputs are natural log probabilities")
    finite = np.isfinite(a) & np.isfinite(b)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        delta = a - b
        delta_p = np.exp(a) - np.exp(b)
    valid = bool(np.all(finite) and np.all(np.isfinite(delta)))
    report = {
        "status": "ok" if valid else "nonfinite_logprob_comparison",
        "delta_convention": "train_minus_infer",
        "n_nonfinite_valid_pairs": int(np.sum(~finite)),
        "delta_logprob_mean": finite_number(np.mean(delta)) if valid else None,
        "abs_delta_logprob": stats(np.abs(delta)) if valid else None,
        "abs_delta_p": stats(np.abs(delta_p)) if valid else None,
        "delta_p_note": "exp may underflow; retain original logprobs",
    }
    full_delta = np.full(mask.shape, np.nan)
    full_delta_p = np.full(mask.shape, np.nan)
    full_delta[mask], full_delta_p[mask] = delta, delta_p
    return report, {"delta_logprob": full_delta, "delta_p": full_delta_p}


def compare(train, infer, kind="hidden", auxiliary=False, chunk_rows=1024):
    validate_pair(train, infer, kind)
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if auxiliary and kind != "hidden":
        raise ValueError("--auxiliary applies only to hidden comparisons")
    if kind == "hidden":
        result, per_token = compare_hidden(train, infer, auxiliary, chunk_rows)
    else:
        result, per_token = compare_logprobs(train, infer)
    result.update({
        "schema_version": 2,
        "kind": kind,
        "n_total": int(train["valid_mask"].size),
        "n_valid": int(np.sum(train["valid_mask"])),
        "values_shape": list(train["values"].shape),
        "train_dtype": str(train["values"].dtype),
        "infer_dtype": str(infer["values"].dtype),
        "metric_dtype": "float64",
        "weighting": "equal_weight_per_valid_token",
    })
    per_token.update({"token_keys": train["token_keys"], "valid_mask": train["valid_mask"]})
    return result, per_token


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("hidden", "logprobs"))
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--infer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--per-token", type=Path)
    parser.add_argument("--auxiliary", action="store_true")
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--path-mode", choices=("omit", "absolute"), default="omit",
                        help="Omit input paths by default; use absolute for local provenance")
    args = parser.parse_args()
    try:
        inputs = {args.train.resolve(), args.infer.resolve()}
        outputs = [args.output.resolve()]
        if args.per_token:
            outputs.append(args.per_token.resolve())
        if len(set(outputs)) != len(outputs) or inputs.intersection(outputs):
            raise ValueError("output paths must be distinct and must not overwrite inputs")
        result, per_token = compare(
            load_snapshot(args.train), load_snapshot(args.infer), args.kind,
            args.auxiliary, args.chunk_rows,
        )
        result["path_mode"] = args.path_mode
        if args.path_mode == "absolute":
            result["train_path"] = str(args.train.resolve())
            result["infer_path"] = str(args.infer.resolve())
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.per_token:
            args.per_token.parent.mkdir(parents=True, exist_ok=True)
            with args.per_token.open("wb") as output:
                np.savez_compressed(output, **per_token)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Comparison failed: {exc}\n")


if __name__ == "__main__":
    main()
