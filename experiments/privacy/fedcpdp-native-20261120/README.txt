FedCPDP native privacy-utility study / 2026-11-20

Scope: unchanged author training and aggregation, frozen five-family homogeneous
LOPO data, 30 targets, seeds 0..9, 50 rounds, 5 local epochs, batch 32.
Conditions: none (no clip/no noise), clip_only (C=1, sigma=0),
sigma 0.1/0.25/0.5/1.0 (C=1). These grid values are our study design.
No verified end-to-end epsilon/delta guarantee. Auxiliary losses and statistics
remain unprotected as in the original implementation. Do not label sigma epsilon.
All data and row splits are identical across conditions. No target-label tuning.
Source-validation MCC chooses threshold; final prescribed model is evaluated.
Report per-project seed mean/std then project-macro summaries, including change
relative to none. Retain native original preprocessing and original LOC rules.

Ubuntu x86_64, Python 3.12 or 3.13; CPU sufficient. Start with two workers.
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
PYTHONHASHSEED=0 .venv/bin/python validate_package.py
PYTHONHASHSEED=0 .venv/bin/python -u run_study.py --workers 2
The controller runs ar1/seed0 at full 50 rounds for every condition, verifies
saved models/predictions, and only then starts the 1800-run formal matrix.
The six pilot runs are reused within that SAME package/environment.
Use --pilot-only to stop after the verified pilot. No historical runs imported.
For a detached run use systemd-run or nohup; save stdout/stderr to a log.

Progress: study_status.json plus conditions/<condition>/status.json.
Each condition stores runs/<target>/<seed>/ predictions, validation predictions,
metrics, console log, checkpoint. summarize.py reloads checkpoints and verifies
predictions/metrics before emitting all_runs.csv/project_summary/family_summary.
Keep condition identity when merging CSV files; their internal condition column
describes threshold mode, not privacy strength.
Stop safely by creating STOP in all six condition directories. Existing native
schedulers checkpoint at round boundaries. Remove STOP files before resuming.
Restart the same command to resume; never change frozen config midway.
After a killed process, verify no related process is alive before removing a
stale study.lock or scheduler.lock. Do not delete running-process locks.

For teammate handoff: supply this exact archive + SHA256, allocate disjoint
conditions/targets/seeds, keep configs and split files unchanged, return all
metrics/predictions/logs and environment freeze; retain checkpoints for checking.
Do not publish raw data/credentials or checkpoints into Git by accident.
