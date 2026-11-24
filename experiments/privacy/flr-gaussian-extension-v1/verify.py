"""Audit a completed FLR privacy matrix and write method-level summaries."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "homogeneous" / "flr-tff-20261111"
CONDITIONS = ("none", "clip_only", "sigma_0p1", "sigma_0p25", "sigma_0p5", "sigma_1p0")
# pandas.DataFrame.to_csv writes six decimal places by default.  Accept only
# the corresponding half-unit rounding error when reproducing saved scores.
CSV_SCORE_ATOL = 5.1e-7


def close(a, b, tol=1e-10):
    if a is None or b is None:
        return a is None and b is None
    return math.isfinite(float(a)) and math.isfinite(float(b)) and abs(float(a) - float(b)) <= tol


def classify(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = ((np.asarray(y) == 0) & ~pred).sum(), ((np.asarray(y) == 0) & pred).sum(), ((np.asarray(y) == 1) & ~pred).sum(), ((np.asarray(y) == 1) & pred).sum()
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    from sklearn.metrics import f1_score, roc_auc_score
    return {"ROC_AUC": float(roc_auc_score(y, p)), "MCC": float((tp * tn - fp * fn) / den) if den else 0.0,
            "Recall": float(tp / (tp + fn)), "G_mean": float(math.sqrt(tp / (tp + fn) * tn / (tn + fp))),
            "F1": float(f1_score(y, pred, zero_division=0)), "threshold": float(threshold)}


def effort(y, p, loc):
    keys = [f"{metric}@{budget}" for metric in ("CostEffect", "Popt") for budget in (5, 10, 20, 40)]
    if not np.isfinite(loc).all() or np.any(loc <= 0) or np.sum(y) == 0:
        return {"reason": "Nonpositive/missing LOC or no positive labels", **dict.fromkeys(keys)}
    ids = np.arange(len(y)); result = {}

    def curve(order, budget):
        x = np.r_[0, np.cumsum(loc[order]) / loc.sum()]
        z = np.r_[0, np.cumsum(y[order]) / y.sum()]
        xx = np.r_[x[x < budget], budget]
        zz = np.interp(xx, x, z)
        return float(zz[-1]), float(np.trapz(zz, xx))

    for budget in (5, 10, 20, 40):
        ce, area = curve(np.lexsort((ids, -p / loc)), budget / 100)
        _, best = curve(np.lexsort((ids, -y / loc)), budget / 100)
        _, worst = curve(np.lexsort((ids, y / loc)), budget / 100)
        result[f"CostEffect@{budget}"] = ce
        result[f"Popt@{budget}"] = (area - worst) / (best - worst) if abs(best - worst) > 1e-12 else None
    return result


def chosen_threshold(values):
    choices = []
    for threshold in np.linspace(0.05, 0.95, 37):
        metrics = [classify(y, p, threshold) for y, p in values]
        choices.append(((np.mean([m["MCC"] for m in metrics]),
                         np.mean([m["Recall"] for m in metrics]), -threshold), threshold))
    return float(max(choices)[1])


def fingerprint(condition, task, target):
    payload = b"".join((HERE / name).read_bytes() for name in ("train.py", "privacy.py", "protocol.json"))
    payload += (BASE / "train_tff.py").read_bytes()
    payload += json.dumps(task, sort_keys=True).encode()
    payload += hashlib.sha256((BASE / "prepared" / f"{target}.npz").read_bytes()).hexdigest().encode()
    return hashlib.sha256(payload + condition.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("formal",), default="formal")
    args = ap.parse_args()
    del args
    cfg = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    tasks = json.loads((BASE / "tasks.json").read_text(encoding="utf-8"))
    result_root = HERE / "runs"
    report_root = result_root / "formal"
    report_root.mkdir(parents=True, exist_ok=True)
    errors = []; rows = []; saturation = {}; edge = {}
    for condition in CONDITIONS:
        sigma = cfg["conditions"][condition]["sigma"]
        for target, task in sorted(tasks.items()):
            arrays = np.load(BASE / "prepared" / f"{target}.npz")
            expected_fp = fingerprint(condition, task, target)
            for seed in range(10):
                label = f"{condition}/{target}/{seed}"; folder = result_root / condition / target / str(seed)
                needed = ["metrics.json", "progress.json", "latest_weights.npz", "predictions.csv"]
                missing = [name for name in needed if not (folder / name).is_file()]
                if missing:
                    errors.append(f"{label}: missing {','.join(missing)}"); continue
                try:
                    metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
                    progress = json.loads((folder / "progress.json").read_text(encoding="utf-8"))
                    if progress.get("status") != "completed" or progress.get("round") != 100:
                        errors.append(f"{label}: incomplete progress")
                    if metrics.get("rounds") != 100 or metrics.get("fingerprint") != expected_fp:
                        errors.append(f"{label}: rounds/fingerprint mismatch")
                    if metrics.get("privacy_condition") != condition or metrics.get("target") != target or metrics.get("seed") != seed:
                        errors.append(f"{label}: result identity mismatch")
                    weights = np.load(folder / "latest_weights.npz")
                    w, b = weights["arr_0"], weights["arr_1"]
                    pred = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
                    y, p = arrays["y_test"], expit(arrays["X_test"] @ w + b).reshape(-1)
                    if not np.array_equal(pred["row_id"].to_numpy(), np.arange(len(y))): errors.append(f"{label}: row ids mismatch")
                    if not np.array_equal(pred["y_true"].to_numpy(), y): errors.append(f"{label}: labels mismatch")
                    if not np.allclose(pred["score"].to_numpy(), p, rtol=0, atol=CSV_SCORE_ATOL): errors.append(f"{label}: target scores cannot be reproduced")
                    values = []
                    for i, source in enumerate(task["sources"]):
                        val_path = folder / f"validation_{source}.csv"
                        if not val_path.is_file(): errors.append(f"{label}: missing validation prediction {source}"); continue
                        val = pd.read_csv(val_path, float_precision="round_trip")
                        vp = expit(arrays[f"V_{i}"] @ w + b).reshape(-1)
                        if not np.array_equal(val["y_true"].to_numpy(), arrays[f"v_{i}"]): errors.append(f"{label}: validation labels mismatch {source}")
                        if not np.allclose(val["score"].to_numpy(), vp, rtol=0, atol=CSV_SCORE_ATOL): errors.append(f"{label}: validation scores mismatch {source}")
                        values.append((arrays[f"v_{i}"], vp))
                    threshold = chosen_threshold(values)
                    if not close(threshold, metrics["study"]["threshold"]): errors.append(f"{label}: threshold mismatch")
                    for mode, expected in (("study", metrics["study"]), ("native", metrics["native"])):
                        actual = classify(y, p, expected["threshold"])
                        for key in ("ROC_AUC", "MCC", "Recall", "G_mean", "F1"):
                            if not close(actual[key], expected[key]): errors.append(f"{label}/{mode}: metric mismatch {key}")
                    actual_effort = effort(y, p, arrays["loc"])
                    saved_effort = metrics.get("effort")
                    for key, value in actual_effort.items():
                        if key == "reason" or saved_effort is None:
                            continue
                        if not close(value, saved_effort.get(key)): errors.append(f"{label}: effort mismatch {key}")
                    sat = float(np.mean((p <= 1e-6) | (p >= 1 - 1e-6))); saturation[label] = sat
                    edge[label] = float(metrics["study"]["threshold"]) in (0.05, 0.95)
                    for mode, report, cutoff in (("study", metrics["study"], metrics["study"]["threshold"]), ("native", metrics["native"], metrics["native"]["threshold"])):
                        rows.append({"experiment_type": cfg["experiment_type"], "method": "FLR", "privacy_condition": condition,
                                     "threshold_mode": mode, "target": target, "family": task["family"], "seed": seed, "rounds": 100,
                                     "clip_norm": 1.0, "noise_multiplier": sigma, "code_commit": cfg["base_repository_commit"],
                                     "config_fingerprint": expected_fp, **{k: report[k] for k in ("ROC_AUC", "MCC", "Recall", "G_mean", "F1")},
                                     **({k: actual_effort.get(k) for k in actual_effort if k != "reason"} if mode == "study" else {}),
                                     "threshold": cutoff, "saturated_probability_fraction": sat})
                except Exception as exc:
                    errors.append(f"{label}: verification exception {type(exc).__name__}: {exc}")
    frame = pd.DataFrame(rows); expected_rows = len(tasks) * 10 * len(CONDITIONS) * 2
    if len(frame) != expected_rows: errors.append(f"expected {expected_rows} rows, got {len(frame)}")
    if not frame.empty:
        study = frame[frame.threshold_mode == "study"]
        metric_cols = [c for c in ("ROC_AUC", "MCC", "Recall", "G_mean", "F1", "CostEffect@20", "Popt@20") if c in study]
        target = study.groupby(["privacy_condition", "target", "family"], as_index=False)[metric_cols].agg(["mean", "std"]).reset_index()
        target.to_csv(report_root / "target_seed_summary.csv", index=False)
        method = []; family_means = []
        for condition, subset in study.groupby("privacy_condition"):
            row_base = {"privacy_condition": condition}
            for metric in metric_cols:
                means = subset.groupby("target")[metric].mean()
                row_base[metric + "_target_macro_mean"] = float(means.mean())
                row_base[metric + "_target_macro_std"] = float(means.std(ddof=1))
                row_base[metric + "_valid_targets"] = int(means.notna().sum())
                if condition != "none":
                    reference = study[study.privacy_condition == "none"].groupby("target")[metric].mean()
                    row_base[metric + "_delta_vs_none"] = float((means - reference).mean())
                if condition not in ("none", "clip_only"):
                    reference = study[study.privacy_condition == "clip_only"].groupby("target")[metric].mean()
                    row_base[metric + "_delta_vs_clip_only"] = float((means - reference).mean())
            method.append(row_base)
        for (condition, family), subset in study.groupby(["privacy_condition", "family"]):
            row_family = {"privacy_condition": condition, "family": family}
            for metric in metric_cols:
                means = subset.groupby("target")[metric].mean()
                row_family[metric + "_project_macro_mean"] = float(means.mean())
                row_family[metric + "_project_macro_std"] = float(means.std(ddof=1))
                row_family[metric + "_valid_targets"] = int(means.notna().sum())
            family_means.append(row_family)
        pd.DataFrame(method).to_csv(report_root / "method_summary.csv", index=False); frame.to_csv(report_root / "run_metrics.csv", index=False)
        pd.DataFrame(family_means).to_csv(report_root / "family_summary.csv", index=False)
    report = {"status": "passed" if not errors else "failed", "expected_tasks": len(tasks) * 10 * len(CONDITIONS),
              "verified_tasks": len(frame) // 2, "verified_rows": len(frame), "error_count": len(errors), "errors": errors[:100],
              "error_list_truncated": len(errors) > 100, "saturated_probability_tasks": sum(v > 0 for v in saturation.values()),
              "max_saturated_probability_fraction": max(saturation.values(), default=0.0), "threshold_edge_count": sum(edge.values())}
    (report_root / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors: raise SystemExit(1)


if __name__ == "__main__":
    main()
