"""Audit a completed ALMITY privacy matrix and write method-level summaries."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.special import expit

# NumPy < 2.0 exposes the same trapezoidal integral as ``trapz``.
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "homogeneous" / "almity-svm-20261114"
sys.path.insert(0, str(BASE))
spec = importlib.util.spec_from_file_location("almity_baseline", BASE / "train.py")
baseline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baseline)


def expected_fingerprint(task: dict, condition: str) -> str:
    payload = b"".join((HERE / name).read_bytes()
                       for name in ("train.py", "privacy.py", "protocol.json"))
    payload += (BASE / "train.py").read_bytes()
    return hashlib.sha256(payload + task["prepared_content_sha256"].encode()
                           + condition.encode()).hexdigest()


def close(a, b, tol=1e-10) -> bool:
    if a is None or b is None:
        return a is b
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def check_metric(actual: dict, expected: dict, label: str, errors: list[str]) -> None:
    for key in ("ROC_AUC", "MCC", "Recall", "G_mean", "F1"):
        if key not in actual or not close(actual[key], expected[key]):
            errors.append(f"{label}: metric mismatch {key}: {actual.get(key)} != {expected.get(key)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["formal"], default="formal")
    args = ap.parse_args()
    del args
    cfg = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    tasks = json.loads((BASE / "tasks.json").read_text(encoding="utf-8"))
    conditions = list(cfg["conditions"])
    seeds = list(range(10))
    errors: list[str] = []
    rows: list[dict] = []
    saturated = {}
    edge_thresholds = {}
    root = HERE / "runs" / "formal"

    for condition in conditions:
        for target, task in sorted(tasks.items()):
            arrays = np.load(BASE / "prepared" / f"{target}.npz")
            if baseline.content_hash(arrays) != task["prepared_content_sha256"]:
                errors.append(f"{target}: prepared data content hash mismatch")
            fp = expected_fingerprint(task, condition)
            for seed in seeds:
                label = f"{condition}/{target}/{seed}"
                folder = root / condition / target / str(seed)
                paths = [folder / name for name in
                         ("metrics.json", "progress.json", "checkpoint.npz",
                          "aggregation.json", "perturbation_diagnostics.json", "predictions.csv")]
                missing = [p.name for p in paths if not p.is_file()]
                if missing:
                    errors.append(f"{label}: missing {','.join(missing)}")
                    continue
                try:
                    metrics = json.loads(paths[0].read_text(encoding="utf-8"))
                    progress = json.loads(paths[1].read_text(encoding="utf-8"))
                    diag = json.loads(paths[4].read_text(encoding="utf-8"))
                    if progress.get("status") != "completed" or progress.get("round") != 100:
                        errors.append(f"{label}: incomplete progress {progress}")
                    if metrics.get("rounds") != 100 or metrics.get("fingerprint") != fp:
                        errors.append(f"{label}: rounds/fingerprint mismatch")
                    if metrics.get("privacy_condition") != condition or metrics.get("seed") != seed:
                        errors.append(f"{label}: result identity mismatch")
                    if diag.get("condition") != condition or len(diag.get("rounds", [])) != 100:
                        errors.append(f"{label}: incomplete perturbation diagnostics")
                    if [d.get("round") for d in diag.get("rounds", [])] != list(range(1, 101)):
                        errors.append(f"{label}: diagnostic round sequence mismatch")
                    if any(len(d.get("clients", [])) != len(task["sources"])
                           for d in diag.get("rounds", [])):
                        errors.append(f"{label}: diagnostic client count mismatch")
                    if condition != "none":
                        for rd in diag["rounds"]:
                            for client in rd["clients"]:
                                if client.get("post_clip_norm", float("inf")) > 1.0 + 1e-10:
                                    errors.append(f"{label}: clipped update exceeds norm C")
                                    break

                    with np.load(paths[2]) as ck:
                        w = ck["w"].copy(); b = float(ck["b"])
                        if int(ck["round"]) != 100 or str(ck["fingerprint"]) != fp:
                            errors.append(f"{label}: checkpoint identity mismatch")
                    ca = float(metrics["calibration"]["slope"])
                    cb = float(metrics["calibration"]["intercept"])
                    pred = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
                    x, y, loc = arrays["X_test"], arrays["y_test"], arrays["loc"]
                    margin = x @ w + b
                    prob = expit(ca * margin + cb)
                    if not np.array_equal(pred["row_id"].to_numpy(), np.arange(len(y))):
                        errors.append(f"{label}: target row ids mismatch")
                    if not np.array_equal(pred["y_true"].to_numpy(), y):
                        errors.append(f"{label}: target labels mismatch")
                    for col, expected in (("margin", margin), ("probability", prob), ("raw_LOC", loc)):
                        if col not in pred or not np.allclose(pred[col].to_numpy(), expected,
                                                             rtol=1e-12, atol=1e-14):
                            errors.append(f"{label}: target {col} cannot be reproduced")
                    if not np.isfinite(prob).all():
                        errors.append(f"{label}: non-finite probability")

                    vals = []
                    for i, source in enumerate(task["sources"]):
                        val_path = folder / f"validation_{source}.csv"
                        if not val_path.is_file():
                            errors.append(f"{label}: missing validation prediction {source}")
                            continue
                        v = pd.read_csv(val_path, float_precision="round_trip")
                        vy = arrays[f"v_{i}"]; vm = arrays[f"V_{i}"] @ w + b
                        vp = expit(ca * vm + cb)
                        if not np.array_equal(v["y_true"].to_numpy(), vy) or not np.allclose(
                                v["margin"].to_numpy(), vm, rtol=1e-12, atol=1e-14) or not np.allclose(
                                v["probability"].to_numpy(), vp, rtol=1e-12, atol=1e-14):
                            errors.append(f"{label}: validation predictions mismatch for {source}")
                        vals.append((vy, vp))
                    if len(vals) == len(task["sources"]):
                        threshold = baseline.chosen_threshold(vals)
                        if not close(threshold, metrics["study"]["threshold"]):
                            errors.append(f"{label}: validation-only threshold mismatch")

                    study = baseline.classification(y, prob, metrics["study"]["threshold"])
                    fixed = baseline.classification(y, prob, 0.5)
                    native = baseline.classification(y, margin, 0.0)
                    check_metric(metrics["study"], study, label + "/study", errors)
                    check_metric(metrics["fixed_probability"], fixed, label + "/fixed", errors)
                    check_metric(metrics["native_margin"], native, label + "/native", errors)
                    effort = baseline.effort(y, prob, loc)
                    for key, val in effort.items():
                        if key not in metrics["effort"] or not close(metrics["effort"][key], val):
                            errors.append(f"{label}: effort metric mismatch {key}")
                    threshold = float(metrics["study"]["threshold"])
                    edge_thresholds[label] = threshold in (0.05, 0.95)
                    saturated[label] = float(np.mean((prob <= 1e-6) | (prob >= 1.0 - 1e-6)))
                    for mode, evaluated, cutoff in (
                        ("study", metrics["study"], threshold),
                        ("fixed_probability", metrics["fixed_probability"], 0.5),
                        ("native_margin", metrics["native_margin"], 0.0),
                    ):
                        rows.append({"experiment_type": metrics["experiment_type"], "method": "ALMITY",
                            "privacy_condition": condition, "threshold_mode": mode, "target": target,
                            "family": task["family"], "seed": seed, "rounds": 100,
                            "clip_norm": 1.0, "noise_multiplier": cfg["conditions"][condition]["sigma"],
                            "code_commit": cfg["base_repository_commit"], "config_fingerprint": fp,
                            **{k: evaluated[k] for k in ("ROC_AUC", "MCC", "Recall", "G_mean", "F1")},
                            **(metrics["effort"] if mode != "native_margin" else
                               {k: None for k in metrics["effort"]}), "threshold": cutoff,
                            "saturated_probability_fraction": saturated[label]})
                except Exception as exc:
                    errors.append(f"{label}: verification exception {type(exc).__name__}: {exc}")

    frame = pd.DataFrame(rows)
    expected_count = len(tasks) * len(seeds) * len(conditions) * 3
    if len(frame) != expected_count:
        errors.append(f"expected {expected_count} unique task rows, got {len(frame)}")
    if not frame.empty and frame.duplicated(
            ["privacy_condition", "target", "seed", "threshold_mode"]).any():
        errors.append("duplicate task identities found")

    if len(frame):
        study_frame = frame[frame.threshold_mode == "study"]
        metric_cols = [c for c in ("ROC_AUC", "MCC", "Recall", "G_mean", "F1",
            "CostEffect@20", "Popt@20") if c in study_frame.columns]
        grouped = study_frame.groupby(["privacy_condition", "target", "family"], as_index=False)[metric_cols]
        target_summary = grouped.agg(["mean", "std"]).reset_index()
        target_summary.to_csv(root / "target_seed_summary.csv", index=False)
        method_rows = []
        for condition, subset in study_frame.groupby("privacy_condition"):
            base = study_frame[(study_frame.privacy_condition == "none")]
            clip = study_frame[(study_frame.privacy_condition == "clip_only")]
            for metric in metric_cols:
                target_means = subset.groupby("target")[metric].mean()
                row = {"privacy_condition": condition, "metric": metric,
                       "target_macro_mean": float(target_means.mean()),
                       "target_macro_std": float(target_means.std(ddof=1)),
                       "valid_targets": int(target_means.notna().sum()),
                       "delta_vs_none": None, "delta_vs_clip_only": None}
                if condition != "none":
                    ref = base.groupby("target")[metric].mean()
                    row["delta_vs_none"] = float((target_means - ref).mean())
                if condition not in ("none", "clip_only"):
                    ref = clip.groupby("target")[metric].mean()
                    row["delta_vs_clip_only"] = float((target_means - ref).mean())
                method_rows.append(row)
        pd.DataFrame(method_rows).to_csv(root / "method_summary.csv", index=False)
        frame.to_csv(root / "run_metrics.csv", index=False)

    report = {"status": "passed" if not errors else "failed", "expected_tasks": expected_count,
        "verified_tasks": len(frame), "error_count": len(errors), "errors": errors[:100],
        "error_list_truncated": len(errors) > 100,
        "saturated_probability_tasks": sum(v > 0 for v in saturated.values()),
        "max_saturated_probability_fraction": max(saturated.values(), default=0.0),
        "threshold_edge_count": sum(edge_thresholds.values()),
        "completed_marker": json.loads((root / "study_progress.json").read_text()).get("status")
            if (root / "study_progress.json").is_file() else "missing"}
    (root / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
