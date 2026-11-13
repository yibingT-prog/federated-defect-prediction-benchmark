# FLR homogeneous experiment — completed, with audit limitations

30 targets × seeds 0–9, 300 models, 100 actual aggregation rounds each. Completed on 2026-11-13. This is a fixed-protocol FLR experiment using native TensorFlow Federated 0.20.0, not a full reproduction of the original paper and not a matched comparison with the separately published FedCPDP run.

**Audit limitation:** 21 of 30 target tasks have identical feature rows in one source client's training set and another client's validation set. The original per-client grouped split did not prevent this cross-client overlap. Same-schema target-feature matches were excluded from sources. Numerical checks passed, but validation-based threshold selection may be affected by the overlap. Do not present this release as a globally leakage-free final benchmark. Exact affected targets and unique overlapping feature-pattern counts are in the [audit](../../../results/flr-homogeneous-20261111/audit.json).

## Scope and protocol

- Within-family LOPO: PROMISE 12, SOFTLAB 5, NASA 5, AEEEM 5, Relink 3. A project is a client; target labels are used only after training.
- Author FedCPDP CSV release, one supplied version per project; these are not necessarily the latest available versions. No public distillation reserve: camel and ar1 remain ordinary projects.
- Reuse fixed source splits (seed 20261110) across all ten model seeds. Approximately 80/20 grouped source train/validation. The seed changes training randomness, not the split.
- Source minmax uses source training data; target minmax uses all unlabeled target features. This is transductive target-feature access. No feature hashing; retain all metrics including LOC as predictors.
- Dense(1, sigmoid), zero initialization, binary crossentropy, SGD; 100 rounds, 100 local epochs, batch 20, shuffle buffer 500, client LR 0.1, server LR 1.0, full participation; sample-weighted aggregation. No extra oversampling or differential-privacy noise.
- Final-round model; primary threshold maximizes source macro MCC (ties: macro Recall, lower threshold); supplementary threshold 0.5. Effort rank is probability/raw LOC; fractional-budget interpolation, truncated normalized Popt. Effort results exist for 13 targets only.

## Files and reproduction

`train_tff.py`, `schedule.py`, `summarize.py`, `prepare.py`, `build_and_start.py`, `smoke_tff.py`, `download_legacy_wheel.py`, configuration and Docker files are byte-preserved executed files. `prepare.py` is the historical preparer and expects the original sibling FedCPDP output tree. **Use the new `prepare_frozen.py` instead**: it reconstructs inputs directly from published split indices and checks all 30 prepared array-content hashes. No FedCPDP training is required.

From the repository root, use the root pinned Python requirements to prepare inputs:

```bash
python experiments/homogeneous/flr-tff-20261111/prepare_frozen.py --archive /absolute/path/Dataset.zip
cd experiments/homogeneous/flr-tff-20261111
python download_legacy_wheel.py
docker build -t flr-tff020:20261111 .
docker run --rm --cpus 2 --memory 3g --mount type=bind,source=/absolute/path/to/this/experiment,target=/work flr-tff020:20261111 python smoke_tff.py
docker run --rm --cpus 2 --memory 3g --mount type=bind,source=/absolute/path/to/this/experiment,target=/work flr-tff020:20261111 python train_tff.py --target ar5 --seed 0 --rounds 2 --pilot
docker run -d --name flr-homogeneous-reproduction --cpus 4 --memory 4g --memory-swap 4g --mount type=bind,source=/absolute/path/to/this/experiment,target=/work flr-tff020:20261111 python schedule.py
```

Replace the bind source with your absolute checkout path. The pinned dataset archive and hash are documented in `../../completed/fedcpdp-paper-lopo-20261110/provenance.json` and the root dataset-preparation script. The JAX wheel is fetched from its official archive; its expected SHA256 is `662bc4a8ccaa3db446c417381685b74c274c173ed00d986531896be7f4277553`. Do not commit wheels, prepared arrays, raw data or checkpoints.

The historical scheduler skips tasks with existing `metrics.json`; incomplete seeds restart from round 1. Use a fresh output tree for modified code/configuration. It does not implement robust fingerprint-based resume gating. TensorFlow 2.8.0 is used because TFF 0.20.0 requires it, despite the author's README listing TF 2.5.0; the dependency audit and installed package freeze are preserved. The Dockerfile pins direct requirements, but a rebuild may resolve different transitive dependencies; compare with `environment.freeze.txt`. Bitwise reproducibility across platforms is not claimed.

`audit_results.py` is a post-run audit added for publication, using root Python requirements. Supply `--run-root`, `--raw-data` and `--output`. It verifies coverage, all metrics, validation thresholds and predictions against final weights, and reports cross-client overlap. It requires trusted local run outputs, which are not redistributed here.

## Provenance and attribution

FLR: Yamamoto et al., *Towards Privacy Preserving Cross Project Defect Prediction with Federated Learning*, SANER 2023, [author repository](https://github.com/posl/SANER2023_CPDPwithFL). `author_provenance.json` identifies consulted author files. Original author files and third-party packages are linked, not vendored in this directory. Our adapter calls the native TFF API and declares protocol deviations.

The historical launcher saved `run_fingerprint.json` **before** changing only `protocol.json.status`. Restoring that status to `frozen_pending_TFF_smoke_test` reproduces its recorded hash with original Windows CRLF bytes. All other listed hashes match. `publication_provenance.json` separately records the actually published bytes. Historical records are retained rather than rewritten.

See [Chinese analysis and figures](../../../results/flr-homogeneous-20261111/README.md). FedCPDP remains under `experiments/completed/fedcpdp-paper-lopo-20261110/` with its own results directory. Do not pool these experiments or interpret their score difference as method superiority.
