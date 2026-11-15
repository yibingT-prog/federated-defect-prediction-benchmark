# ALMITY linear SVM paper reimplementation — completed release

This is a declared paper-based reimplementation, not author code or a claim of matching paper tables. The 300-run server experiment completed and was verified on 2026-11-14. README.md preserves the original pre-run handoff notes; its statements that Linux execution and GitHub publication were pending are historical, superseded by this release and the result records.

Training code, protocol, tasks and reference splits are byte-identical to the executed package. Hyperparameters are our study settings. The original package included raw/prepared data and sample outputs; this source release retrieves and reconstructs data instead. Do not use the historical README's verify_package.py command in this source release.

From repository root:

```bash
python -m pip install -r experiments/homogeneous/almity-svm-20261114/requirements.txt
python scripts/prepare_almity_data.py --download
python experiments/homogeneous/almity-svm-20261114/prepare.py
python experiments/homogeneous/almity-svm-20261114/check.py
python experiments/homogeneous/almity-svm-20261114/schedule.py --workers 4
python experiments/homogeneous/almity-svm-20261114/summarize.py
```

Data retrieval also accepts `--archive /path/to/Dataset.zip`; all 30 CSV hashes are checked before writing. Third-party datasets are not relicensed here. prepare.py checks generated reference splits; check.py verifies prepared content hashes against the frozen tasks. Use the pinned dependencies. For published results, `python scripts/verify_almity_publication.py` recalculates calibrated and native metrics without raw data or model checkpoints.

New runtime outputs stay in the experiment folder. The [published results](../../../results/almity-homogeneous-20261114/README.md) remain separate from FLR, native FedCPDP, and previous server variants. See protocol.json for calibration reuse, target access and privacy limitations.
