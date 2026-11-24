#!/usr/bin/env bash
set -u
ROOT=/root/federated-defect-prediction-benchmark
P="$ROOT/experiments/privacy/flr-gaussian-extension-v1/runs/_scheduler/pilot/progress.json"
while true; do
  if grep -q '"completed": 12' "$P" 2>/dev/null; then
    if /root/miniconda3/envs/flr-tff020/bin/python - "$P" <<'PY'
import json, pathlib, sys
p = json.loads(pathlib.Path(sys.argv[1]).read_text())
ok = p.get("total") == 12 and p.get("completed") == 12 and p.get("failed") == 0
for record in p.get("records", []):
    out = pathlib.Path(record["output"])
    state = json.loads(out.read_text()) if out.exists() else {}
    ok = ok and state.get("status") == "pilot_passed" and state.get("round") == 100
raise SystemExit(0 if ok else 1)
PY
    then
      exec /root/miniconda3/bin/conda run --no-capture-output -n flr-tff020 \
        python "$ROOT/experiments/privacy/flr-gaussian-extension-v1/run_study.py" \
        --stage formal --workers 24 --rounds 100
    fi
    echo "pilot failed; formal run withheld" >&2
    exit 1
  fi
  sleep 20
done
