"""Collect document evidence without changing historical experiments."""
from pathlib import Path
import collections
import csv
import hashlib
import json
import tarfile
import math
import statistics
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "实验报告与交接材料"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    OUT.mkdir(exist_ok=True)
    report = {"scope": "document evidence audit; does not replace model verification", "methods": {}}
    for method, base_name in [("FLR", "flr-tff-20261111"), ("ALMITY", "almity-svm-20261114")]:
        root = ROOT / f"experiments/privacy/{method.lower()}-gaussian-extension-v1"
        base = ROOT / "experiments/homogeneous" / base_name
        tasks = json.loads((base / "tasks.json").read_text(encoding="utf-8"))
        rows = list(csv.DictReader((root / "runs/formal/run_metrics.csv").open(encoding="utf-8-sig", newline="")))
        study = [r for r in rows if r["threshold_mode"] == "study"]
        conditions = list(json.loads((root / "protocol.json").read_text())["conditions"])
        ids = {(r["privacy_condition"], r["target"], int(r["seed"])) for r in study}
        expected = {(c, t, s) for c in conditions for t in tasks for s in range(10)}
        diagnostics = []
        for c in conditions:
            rs = [r for r in study if r["privacy_condition"] == c]
            sat = [float(r["saturated_probability_fraction"]) for r in rs]
            diagnostics.append({"condition": c, "tasks": len(rs), "saturated_tasks": sum(x > 0 for x in sat),
                                "max_saturation_fraction": max(sat),
                                "threshold_edges": sum(abs(float(r["threshold"]) - .05) < 1e-10 or abs(float(r["threshold"]) - .95) < 1e-10 for r in rs)})
        run_root = root / ("runs" if method == "FLR" else "runs/formal")
        invalid = []
        calibration = collections.defaultdict(list)
        for c, t, s in sorted(expected):
            folder = run_root / c / t / str(s)
            p = json.loads((folder / "progress.json").read_text())
            m = json.loads((folder / "metrics.json").read_text())
            if p.get("status") != "completed" or p.get("round") != 100 or m.get("rounds") != 100:
                invalid.append(f"{c}/{t}/{s}")
            if "calibration" in m:
                calibration[c].append(float(m["calibration"]["slope"]))
        families = collections.defaultdict(list)
        for t, task in tasks.items():
            families[task["family"]].append(t)
        valid_effort = sorted({r["target"] for r in study if r.get("CostEffect@20") not in (None, "")})
        content_checks = []
        split_checks = []
        dtypes = set()
        for target in tasks:
            with np.load(base / "prepared" / (target + ".npz")) as arrays:
                h = hashlib.sha256()
                for key in sorted(arrays.files):
                    a = arrays[key]
                    h.update(key.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
                expected_content = (json.loads((base / "prepared_content_hashes.json").read_text())[target]
                                    if method == "FLR" else tasks[target]["prepared_content_sha256"])
                content_checks.append(h.hexdigest() == expected_content)
                dtypes.add(str(arrays["X_test"].dtype))
            flr_split = ROOT / "experiments/homogeneous/flr-tff-20261111/splits" / (target + "_splits.json")
            almity_split = ROOT / "experiments/homogeneous/almity-svm-20261114/split_reference" / (target + "_splits.json")
            split_checks.append(json.loads(flr_split.read_text()) == json.loads(almity_split.read_text()))
        summary_rows = list(csv.DictReader((root / "runs/formal/method_summary.csv").open(newline="", encoding="utf-8-sig")))
        summary_errors = []
        for cond in conditions:
            for metric in ("ROC_AUC", "MCC", "Recall", "G_mean", "F1", "CostEffect@20", "Popt@20"):
                values = []
                for target in tasks:
                    v = [float(r[metric]) for r in study if r["privacy_condition"] == cond and r["target"] == target and r[metric] not in (None, "")]
                    if v:
                        values.append(statistics.mean(v))
                saved = next(r for r in summary_rows if r["privacy_condition"] == cond and (method == "FLR" or r["metric"] == metric))
                mean_key, std_key = ((metric + "_target_macro_mean", metric + "_target_macro_std") if method == "FLR" else ("target_macro_mean", "target_macro_std"))
                if not math.isclose(statistics.mean(values), float(saved[mean_key]), abs_tol=1e-12) or not math.isclose(statistics.stdev(values), float(saved[std_key]), abs_tol=1e-12):
                    summary_errors.append(f"{cond}/{metric}")
        info = {"rows": len(rows), "study_rows": len(study), "unique_models": len(ids),
                "missing": sorted(expected - ids), "extra": sorted(ids - expected),
                "duplicate_study_rows": len(study) - len(ids), "invalid_progress": invalid,
                "prepared_content_hash_passed": sum(content_checks), "prepared_dtypes": sorted(dtypes),
                "cross_method_frozen_split_matches": sum(split_checks), "summary_recalculation_errors": summary_errors,
                "threshold_modes": sorted({r["threshold_mode"] for r in rows}),
                "families": dict(families), "valid_effort_targets": valid_effort,
                "na_effort_targets": sorted(set(tasks) - set(valid_effort)),
                "verification": json.loads((root / "runs/formal/verification.json").read_text()),
                "diagnostics": diagnostics,
                "source_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in root.iterdir() if p.is_file()},
                "calibration": {c: {"negative_slope_tasks": sum(x < 0 for x in v), "min_slope": min(v), "max_slope": max(v)} for c, v in calibration.items()}}
        report["methods"][method] = info
    archive = ROOT / "服务器迁移备份_2026-11-24/project-and-results.tar.gz"
    prefix = "federated-defect-prediction-benchmark/"
    counts = collections.Counter()
    selected = {}
    source_matches = {}
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            name = member.name.removeprefix(prefix)
            parts = name.split("/")
            if name.startswith("experiments/privacy/") and "extension-v1" in name:
                method_root = "/".join(parts[:3])
                rel = "/".join(parts[3:])
                if rel.startswith("runs/") and parts[-1] in {"metrics.json", "latest_weights.npz", "checkpoint.npz", "perturbation_diagnostics.json"}:
                    counts[method_root + "/" + parts[-1]] += 1
                if len(parts) == 4 or (len(parts) <= 7 and parts[-1] in {"verification.json", "pilot_verified.log", "formal_guard.log", "environment.actual.txt"}):
                    data = tar.extractfile(member).read()
                    if len(parts) == 4:
                        local = ROOT / name
                        source_matches[name] = {"archive_sha256": hashlib.sha256(data).hexdigest(), "local_equal": local.is_file() and local.read_bytes() == data}
                    if parts[-1].endswith((".json", ".log", ".txt")) and len(data) < 20000:
                        selected[name] = data.decode("utf-8", errors="replace")
                elif len(parts) <= 8 and any(k in parts[-1].lower() for k in ["alignment", "equivalence", "runtime", "environment", "freeze", "provenance", "preflight"]):
                    if member.size < 30000:
                        selected[name] = tar.extractfile(member).read().decode("utf-8", errors="replace")
                if "pilot" in rel and parts[-1] == "progress.json" and "pre_alignment" not in rel:
                    counts[method_root + "/pilot_progress"] += 1
                    selected[name] = tar.extractfile(member).read().decode("utf-8", errors="replace")
            elif name in {".git/HEAD", ".git/packed-refs", ".git/logs/HEAD", "requirements.txt"}:
                selected[name] = tar.extractfile(member).read().decode("utf-8", errors="replace")[-10000:]
    report["archive_counts"] = dict(counts)
    report["archive_source_matches"] = source_matches
    report["archive_selected_records"] = selected
    report["archives"] = {str(p.relative_to(ROOT)).replace("\\", "/"): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in [archive, ROOT / "服务器迁移备份_2026-11-24/flr-tff020-environment.tar.gz", ROOT / "FLR_ALMITY_隐私实验交付_最终版.zip"]}
    (OUT / "文档依据核对.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"methods": {k: {s: v[s] for s in ("rows", "unique_models", "missing", "duplicate_study_rows", "invalid_progress", "diagnostics", "calibration")} for k,v in report["methods"].items()}, "archive_counts": dict(counts), "selected_records": list(selected), "archive_source_matches": source_matches}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
