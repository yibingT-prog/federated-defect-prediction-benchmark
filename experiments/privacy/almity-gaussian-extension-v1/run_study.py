"""Sequential, resumable formal ALMITY privacy matrix runner."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["formal"], required=True)
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args()

    base = HERE.parents[1] / "homogeneous" / "almity-svm-20261114"
    tasks = json.loads((base / "tasks.json").read_text(encoding="utf-8"))
    protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
    targets = args.targets or sorted(tasks)
    seeds = args.seeds or list(range(10))
    conditions = list(protocol["conditions"])
    unknown = sorted(set(targets) - set(tasks))
    if unknown or any(seed not in range(10) for seed in seeds):
        raise ValueError(f"invalid targets/seeds: {unknown}, {seeds}")

    root = HERE / "runs" / "formal"
    state_path = root / "study_progress.json"
    state = {"status": "running", "total": len(targets) * len(seeds) * len(conditions),
             "completed": 0, "current": None, "started_at": time.time()}
    save(state_path, state)
    log_root = root / "logs"
    for condition in conditions:
        for target in targets:
            for seed in seeds:
                state["current"] = {"condition": condition, "target": target, "seed": seed}
                save(state_path, state)
                log = log_root / condition / target / f"{seed}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                env = os.environ.copy()
                env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                            "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
                cmd = [sys.executable, str(HERE / "train.py"), "--target", target,
                       "--seed", str(seed), "--condition", condition, "--stage", "formal"]
                with log.open("w", encoding="utf-8") as out:
                    result = subprocess.run(cmd, cwd=HERE.parents[2], env=env,
                                            stdout=out, stderr=subprocess.STDOUT, check=False)
                if result.returncode:
                    state.update(status="failed", failed=state["current"], exit_code=result.returncode,
                                 finished_at=time.time())
                    save(state_path, state)
                    raise SystemExit(f"task failed: {state['failed']}; see {log}")
                state["completed"] += 1
                state["last_completed"] = state["current"]
                state["current"] = None
                save(state_path, state)
                print(f"{state['completed']}/{state['total']} {condition} {target} seed={seed}", flush=True)
    state.update(status="completed", finished_at=time.time(), current=None)
    save(state_path, state)


if __name__ == "__main__":
    main()
