"""Bounded, resumable FLR pilot/formal scheduler."""
import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONDITIONS = ("none", "clip_only", "sigma_0p1", "sigma_0p25", "sigma_0p5", "sigma_1p0")


def tasks(stage, targets, seeds):
    return [(condition, target, seed) for condition in CONDITIONS
            for target in targets for seed in seeds]


def run_one(item, stage, rounds, log_root):
    condition, target, seed = item
    pilot = stage == "pilot"
    out = HERE / ("pilot" if pilot else "runs") / condition / target / str(seed)
    expected = out / ("progress.json" if pilot else "metrics.json")
    if expected.exists():
        try:
            existing = json.loads(expected.read_text(encoding="utf-8"))
            valid = (existing.get("status") == "pilot_passed" and
                     existing.get("round") == rounds) if pilot else (
                     existing.get("method") == "FLR" and
                     existing.get("privacy_condition") == condition and
                     existing.get("target") == target and
                     existing.get("seed") == seed and
                     existing.get("rounds") == rounds)
            if valid:
                return {"condition": condition, "target": target, "seed": seed,
                        "status": "completed", "returncode": 0, "seconds": 0,
                        "skipped_existing": True, "output": str(expected)}
        except (OSError, ValueError):
            pass
    log = log_root / condition / target
    log.mkdir(parents=True, exist_ok=True)
    log_path = log / (f"{seed}.log")
    cmd = [sys.executable, str(HERE / "train.py"), "--target", target,
           "--seed", str(seed), "--condition", condition, "--rounds", str(rounds)]
    if pilot:
        cmd.append("--pilot")
    env = os.environ.copy()
    env.setdefault("TF_NUM_INTRAOP_THREADS", "1")
    env.setdefault("TF_NUM_INTEROP_THREADS", "1")
    start = time.time()
    with log_path.open("ab") as stream:
        stream.write(("\n=== START %s %s %s ===\n" % item).encode())
        result = subprocess.run(cmd, cwd=HERE.parent.parent.parent, stdout=stream,
                                stderr=subprocess.STDOUT, env=env)
    ok = result.returncode == 0 and expected.exists()
    return {"condition": condition, "target": target, "seed": seed,
            "status": "completed" if ok else "failed", "returncode": result.returncode,
            "seconds": round(time.time() - start, 2), "log": str(log_path),
            "output": str(expected)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("pilot", "formal"), required=True)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=None)
    ap.add_argument("--scheduler-name", default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=100)
    a = ap.parse_args()
    task_file = HERE.parents[1] / "homogeneous" / "flr-tff-20261111" / "tasks.json"
    all_tasks = json.loads(task_file.read_text(encoding="utf-8"))
    targets = a.targets or list(all_tasks)
    seeds = a.seeds if a.seeds is not None else list(range(10))
    unknown = sorted(set(targets) - set(all_tasks))
    if unknown:
        raise ValueError("unknown targets: " + ", ".join(unknown))
    if a.workers < 1:
        raise ValueError("workers must be positive")
    selected_conditions = a.conditions or CONDITIONS
    work = [(condition, target, seed) for condition in selected_conditions
            for target in targets for seed in seeds]
    scheduler_name = a.scheduler_name or a.stage
    root = HERE / "runs" / "_scheduler" / scheduler_name
    root.mkdir(parents=True, exist_ok=True)
    progress = root / "progress.json"
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        futures = {pool.submit(run_one, item, a.stage, a.rounds, root / "logs"): item for item in work}
        for future in concurrent.futures.as_completed(futures):
            record = future.result()
            records.append(record)
            progress.write_text(json.dumps({"stage": a.stage, "total": len(work),
                                            "completed": sum(r["status"] == "completed" for r in records),
                                            "failed": sum(r["status"] == "failed" for r in records),
                                            "records": records}, indent=2), encoding="utf-8")
            print(record["status"], record["condition"], record["target"], record["seed"],
                  record["seconds"], flush=True)
    failed = [r for r in records if r["status"] != "completed"]
    print(json.dumps({"stage": a.stage, "total": len(work), "failed": len(failed)}, indent=2))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
