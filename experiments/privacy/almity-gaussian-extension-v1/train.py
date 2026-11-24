"""ALMITY privacy extension; keeps the published reimplementation intact."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "homogeneous" / "almity-svm-20261114"
sys.path.insert(0, str(BASE))
import train as baseline  # noqa: E402
from privacy import condition, perturb_update  # noqa: E402


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def target_seed(target):
    return int.from_bytes(hashlib.sha256(target.encode()).digest()[:4], "little")


def fingerprint(task, cond):
    files = [HERE / "train.py", HERE / "privacy.py", HERE / "protocol.json"]
    payload = b"".join(p.read_bytes() for p in files)
    payload += (BASE / "train.py").read_bytes()
    return hashlib.sha256(payload + task["prepared_content_sha256"].encode() + cond.encode()).hexdigest()


def execute(target, seed, cond_name, smoke=False, max_rounds=None, stage="formal"):
    cfg = json.loads((HERE / "protocol.json").read_text())
    task = json.loads((BASE / "tasks.json").read_text())[target]
    params = cfg["training"]
    cond = condition(cond_name)
    if stage not in {"pilot", "formal"}:
        raise ValueError("stage must be pilot or formal")
    folder = HERE / "runs"
    if stage == "formal":
        folder = folder / "formal"
    folder = folder / cond_name / target / str(seed)
    folder.mkdir(parents=True, exist_ok=True)
    arrays = np.load(BASE / "prepared" / (target + ".npz"))
    if baseline.content_hash(arrays) != task["prepared_content_sha256"]:
        raise ValueError("prepared data hash mismatch")
    fp = fingerprint(task, cond_name)
    if (folder / "metrics.json").exists():
        old = json.loads((folder / "metrics.json").read_text())
        if old.get("fingerprint") == fp and old.get("rounds") == params["rounds"]:
            print("SKIP completed", cond_name, target, seed, flush=True); return
    clients = [(arrays[f"X_{i}"], arrays[f"y_{i}"]) for i in range(len(task["sources"]))]
    agg, attributes = baseline.weights([y for _, y in clients])
    baseline.save(folder / "aggregation.json", dict(clients=task["sources"], **attributes,
        auxiliary_statistics_protected=False))
    w = np.zeros(clients[0][0].shape[1]); b = 0.0; completed = 0
    ck = folder / "checkpoint.npz"
    if ck.exists():
        with np.load(ck) as state:
            if str(state["fingerprint"]) != fp: raise ValueError("checkpoint fingerprint mismatch")
            w = state["w"].copy(); b = float(state["b"]); completed = int(state["round"])
    rounds = min(params["rounds"], 2 if smoke else (max_rounds or params["rounds"]))
    diagnostics = []
    diagnostics_path = folder / "perturbation_diagnostics.json"
    if completed and diagnostics_path.exists():
        previous = json.loads(diagnostics_path.read_text())
        if previous.get("condition") != cond_name:
            raise ValueError("diagnostic condition mismatch")
        diagnostics = previous.get("rounds", [])
        if len(diagnostics) > completed:
            raise ValueError("diagnostics exceed checkpoint round")
    for r in range(completed, rounds):
        local = []
        round_diag = []
        for i, (x, y) in enumerate(clients):
            lw = w.copy(); lb = b
            train_rng = np.random.default_rng(np.random.SeedSequence([seed, r, i]))
            for _ in range(params["local_epochs"]):
                order = train_rng.permutation(len(y))
                for j in range(0, len(y), params["batch_size"]):
                    ix = order[j:j + params["batch_size"]]
                    gw, gb = baseline.gradient(lw, lb, x[ix], y[ix], params["l2"])
                    lw -= params["lr"] * gw; lb -= params["lr"] * gb
            if cond.name == "none":
                # Preserve the original path exactly when perturbation is disabled.
                local.append((lw, lb)); round_diag.append({"pre_norm": float(np.linalg.norm(np.r_[lw-w, lb]))})
            else:
                noise_rng = np.random.default_rng(np.random.SeedSequence([seed, target_seed(target), r, i, 991]))
                dw, db, diag = perturb_update(lw - w, lb - b, cond, noise_rng)
                local.append((w + dw, b + float(db))); round_diag.append(diag)
        w = sum(a * v[0] for a, v in zip(agg, local)); b = float(sum(a * v[1] for a, v in zip(agg, local)))
        if not np.isfinite(w).all() or not np.isfinite(b): raise FloatingPointError("non-finite model")
        diagnostics.append({"round": r + 1, "clients": round_diag})
        with (folder / "checkpoint.tmp").open("wb") as out:
            np.savez(out, w=w, b=b, round=r + 1, fingerprint=fp)
        (folder / "checkpoint.tmp").replace(ck)
        baseline.save(folder / "progress.json", {"status": "running", "round": r + 1,
            "rounds": params["rounds"], "target": target, "seed": seed, "privacy_condition": cond_name})
        if (r + 1) % 10 == 0: print(cond_name, target, seed, r + 1, flush=True)
    baseline.save(diagnostics_path, {"condition": cond_name, "rounds": diagnostics})
    if smoke or rounds < params["rounds"]:
        baseline.save(folder / "progress.json", {"status": "smoke_passed", "round": rounds,
            "target": target, "seed": seed, "test_scored": False}); return
    margins = [arrays[f"V_{i}"] @ w + b for i in range(len(clients))]
    labels = [arrays[f"v_{i}"] for i in range(len(clients))]
    calibration = LogisticRegression(C=1., solver="lbfgs", max_iter=1000, random_state=seed).fit(
        np.concatenate(margins).reshape(-1, 1), np.concatenate(labels))
    ca = float(calibration.coef_[0, 0]); cb = float(calibration.intercept_[0])
    vals = [(y, expit(ca * z + cb)) for y, z in zip(labels, margins)]
    threshold = baseline.chosen_threshold(vals)
    raw = arrays["X_test"] @ w + b; p = expit(ca * raw + cb); y = arrays["y_test"]; loc = arrays["loc"]
    report = {"experiment_type": cfg["experiment_type"], "method": "ALMITY", "privacy_condition": cond_name,
        "target": target, "family": task["family"], "seed": seed, "rounds": rounds, "fingerprint": fp,
        "study": {"threshold": threshold, **baseline.classification(y, p, threshold)},
        "fixed_probability": {"threshold": .5, **baseline.classification(y, p, .5)},
        "native_margin": {"threshold": 0., **baseline.classification(y, raw, 0.)},
        "effort": baseline.effort(y, p, loc), "calibration": {"slope": ca, "intercept": cb,
        "fit": "pooled source validation only"}, "auxiliary_statistics_protected": False}
    pd.DataFrame({"row_id": np.arange(len(y)), "y_true": y, "margin": raw,
                  "probability": p, "raw_LOC": loc}).to_csv(folder / "predictions.csv", index=False)
    for i, name in enumerate(task["sources"]):
        pd.DataFrame({"y_true": labels[i], "margin": margins[i], "probability": vals[i][1]}).to_csv(
            folder / ("validation_" + name + ".csv"), index=False)
    baseline.save(folder / "metrics.json", report)
    baseline.save(folder / "progress.json", {"status": "completed", "round": rounds, "target": target, "seed": seed})
    print("COMPLETED", cond_name, target, seed, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True); ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--condition", required=True); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--max-rounds", type=int)
    ap.add_argument("--stage", choices=["pilot", "formal"], default="formal")
    a = ap.parse_args()
    if not 0 <= a.seed <= 9: raise ValueError("seed must be 0..9")
    execute(a.target, a.seed, a.condition, a.smoke, a.max_rounds, a.stage)


if __name__ == "__main__": main()
