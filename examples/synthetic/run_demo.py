#!/usr/bin/env python3
"""Generate synthetic evidence for metrics, replay and controlled intervention."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "skills" / "rl-train-infer-diagnosis" / "scripts"))
from compare_tensors import compare  # noqa: E402


def snapshot(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "values": values,
        "token_keys": np.asarray([f"synthetic:{i}" for i in range(len(values))]),
        "valid_mask": np.ones(len(values), dtype=bool),
    }


def metric(a, b, kind="hidden"):
    return compare(snapshot(a), snapshot(b), kind=kind, auxiliary=kind == "hidden")


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def save_pair(directory, a, b):
    directory.mkdir(parents=True)
    np.savez_compressed(directory / "train.npz", **snapshot(a))
    np.savez_compressed(directory / "infer.npz", **snapshot(b))
    report, per_token = metric(a, b)
    write_json(directory / "metrics.json", report)
    np.savez_compressed(directory / "per_token.npz", **per_token)
    return report


def source_train(x):
    return x.copy()


def source_infer(x):
    return x + np.asarray([0.0, 0.02])


def amplifier(x):
    return x * np.asarray([1.0, 20.0])


def target_logprob(logits):
    # Stable full two-class log-softmax; target class is always 1.
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    return shifted[:, 1] - np.log(np.sum(np.exp(shifted), axis=1))


def endpoint(x, infer_source):
    train = target_logprob(amplifier(source_train(x)))
    infer = target_logprob(amplifier(infer_source(x)))
    return metric(train, infer, kind="logprobs")[0]["abs_delta_logprob"]["mean"]


def run(output):
    # Refuse overwrites: every run gets its own evidence directory.
    output.mkdir(parents=True, exist_ok=False)
    scale = save_pair(output / "scale", [[1, 2], [3, 4]], [[2, 4], [3, 4]])
    x = np.asarray([[1.0, 0.0], [2.0, 0.0], [1.5, 0.0]])
    xt, xi = source_train(x), source_infer(x)
    source = save_pair(output / "source", xt, xi)
    yt, yi = amplifier(xt), amplifier(xi)
    a, b = amplifier(xt), amplifier(xt)
    c, d = amplifier(xi), amplifier(xi)
    amplified = save_pair(output / "amplifier", yt, yi)
    pairs = {"R_T": (a, yt), "R_I": (d, yi), "S_T": (a, b), "S_I": (c, d),
             "C_in": (xt, xi), "P_T": (a, c), "P_I": (b, d), "C_natural": (a, d)}
    replay = {name: metric(left, right)[0] for name, (left, right) in pairs.items()}
    write_json(output / "replay.json", replay)
    for name, values in {"a": a, "b": b, "c": c, "d": d}.items():
        np.savez_compressed(output / f"replay_{name}.npz", **snapshot(values))

    chain = output / "chain"
    chain.mkdir()
    np.savez_compressed(chain / "train_logprobs.npz", **snapshot(target_logprob(a)))
    np.savez_compressed(chain / "infer_logprobs.npz", **snapshot(target_logprob(d)))
    extra = np.asarray([[0.8, 0.001], [1.2, -0.002]])
    baseline = endpoint(x, source_infer)
    intervention = endpoint(x, source_train)
    revert = endpoint(x, source_infer)
    extra_baseline = endpoint(extra, source_infer)
    extra_intervention = endpoint(extra, source_train)
    summary = {
        "data_origin": "synthetic",
        "engine_integration_tested": False,
        "scope": "Two-dimensional NumPy source and amplifier; not an RL model",
        "scale_only": {
            "cosine_mean": scale["cosine_mean"],
            "relative_l2_mean": scale["auxiliary"]["relative_l2_train_denominator"]["mean"],
            "norm_ratio_mean": scale["auxiliary"]["infer_over_train_norm"]["mean"],
        },
        "source_cosine": source["cosine_mean"],
        "amplified_cosine": amplified["cosine_mean"],
        "source_G": 1.0 - source["cosine_mean"],
        "amplifier_G": source["cosine_mean"] - amplified["cosine_mean"],
        "amplifier_replay_cosine": {k: v["cosine_mean"] for k, v in replay.items()},
        "endpoint": {
            "metric": "mean_abs_delta_target_logprob",
            "baseline": baseline,
            "intervention": intervention,
            "revert": revert,
            "reduction": 1.0 - intervention / baseline,
            "extra_baseline": extra_baseline,
            "extra_intervention": extra_intervention,
        },
    }
    write_json(output / "summary.json", summary)
    write_json(output / "run_manifest.json", {
        "data_origin": "synthetic", "dtype": "float64", "n_valid": len(x),
        "weight_snapshot": "not_applicable", "state": "stateless",
        "source_difference": "infer adds [0, 0.02]",
        "shared_amplifier": "elementwise multiply by [1, 20]",
        "score": "two-class raw log-softmax, target index 1",
        "intervention": "replace only source_infer with source_train on the current input",
        "comparison_atol": 1e-12, "comparison_rtol": 1e-12,
        "numpy_version": np.__version__,
    })
    (output / "report.md").write_text(
        "# 合成诊断示例结果\n\n"
        "范围：无状态二维 NumPy 运算；所有数据为合成，未运行 RL 模型或真实引擎。\n\n"
        "纯幅值案例：cosine 约为 1，但第一个 token 的 relative L2=1、norm ratio=2。\n\n"
        "链路案例：推理侧 source 增加 [0, 0.02]；两端 amplifier 都乘以 [1, 20]。"
        "amplifier 的同输入重放相等，但自然输入差异增大；误差来源位于 source。\n\n"
        f"source cosine={source['cosine_mean']:.12f}，"
        f"amplifier cosine={amplified['cosine_mean']:.12f}。\n\n"
        "| 实验 | mean abs delta logprob |\n|---|---:|\n"
        f"| baseline | {baseline:.12f} |\n"
        f"| 仅修复 source | {intervention:.12f} |\n"
        f"| 撤销修复 | {revert:.12f} |\n"
        f"| 额外样本 baseline | {extra_baseline:.12f} |\n"
        f"| 额外样本仅修复 source | {extra_intervention:.12f} |\n\n"
        "这些结果只说明已知合成函数的数值关系，不自动生成真实场景主因状态。"
        "逐 token 数值见各目录的 per_token.npz；四组重放及幅值误差见 replay.json。\n",
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = run(args.output)
    except OSError as exc:
        parser.exit(2, f"Demo failed: {exc}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
