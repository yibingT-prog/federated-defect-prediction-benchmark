# Federated Defect Prediction Benchmark

Reproducible experiments comparing complete federated learning methods for software defect prediction. Native classifiers and core method mechanisms are preserved. The repository is intended to cover homogeneous features, heterogeneous features, privacy effects, and sensitivity analysis.

**Current release:** one completed FedCPDP standalone experiment (30 projects, 90 seed models, 30 ensembles). Homogeneous comparisons, the final heterogeneous comparison, privacy sweeps, and sensitivity experiments are **not implemented or completed yet**. The completed LOPO run includes same-family sources and must not be described as the final cross-family-only RQ2 benchmark.

## Repository layout

```text
experiments/
  completed/fedcpdp-paper-lopo-20261110/  # immutable executed source and protocol
    run.py, check.py, monitor.py
    protocol.json, manifest.json, provenance.json
    vendor/fed_cpdp.py, vendor/LICENSE
  homogeneous/                         # RQ1 scope and future experiments
  heterogeneous/                       # RQ2 scope and admission requirements
  privacy/                             # RQ3 scope, not yet executed
  sensitivity/                         # RQ4 scope, not yet executed
datasets/                              # provenance and local preparation policy
results/fedcpdp-paper-lopo-20261110/     # verified project-level metrics
scripts/                               # dataset preparation and integrity checks
tests/                                 # small metric and split correctness checks
docs/                                  # experiment design and reproduction notes
.github/workflows/                     # data-free CI checks
```

## Reproduce the completed run

Python 3.13.5 was used on Windows. Other platforms can execute the code, but native Python hashing and numerical libraries mean bitwise cross-platform equivalence is not promised.

```bash
python -m venv .venv
# Activate .venv using your operating system's shell.
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/verify_snapshot.py
python scripts/prepare_fedcpdp_data.py --archive /path/to/Dataset.zip
python experiments/completed/fedcpdp-paper-lopo-20261110/check.py
python experiments/completed/fedcpdp-paper-lopo-20261110/run.py --workers 4
```

Alternatively, `python scripts/prepare_fedcpdp_data.py --download` retrieves the pinned upstream archive explicitly. Consult [dataset provenance](datasets/README.md) before using third-party data. Dataset files are not redistributed in this repository.

For a one-round pipeline check, add `--target CM1 --smoke` to `run.py`. It is not a completed experiment. To monitor and validate an active full run, execute `monitor.py` in the same experiment directory from another terminal. The historical monitor validates local trusted pickle checkpoints; never use it with untrusted checkpoint files.

Training saves a checkpoint after each round and resumes with the same command. To stop at round boundaries, create an empty `STOP` file in the experiment directory. Remove that file before resuming. Changing source, protocol, data manifest, or upstream code invalidates checkpoint fingerprints. Historical scheduler status during a STOP request is not a dedicated paused state; inspect the STOP file and logs.

## Completed experiment

- Five families: NASA, SOFTLAB, AEEEM, Relink, and a specific PROMISE collection; 30 project files, 15,901 input rows.
- Leave one entire project out; other 29 projects are available sources.
- Exclude same-schema source feature rows identical to target rows; group identical source features during train/validation splitting.
- Approximately 80/20 source training/validation with a fixed grouped split; target labels never select training parameters, rounds, or thresholds. Native target feature normalization is transductive.
- 50 rounds, 5 local epochs, batch size 32; seeds 42, 43, 44, with separate model and ensemble reporting.
- ROC-AUC, Recall, MCC, G-mean, CostEffect and truncated Popt at 5/10/20/40% LOC budgets. Study MCC and paper-style F1 threshold rules are both saved.
- All 90 models and 30 ensembles completed and were verified. Effort metrics cover 13 projects; 17 contain invalid/non-positive LOC and are explicitly N/A.

See [frozen protocol](experiments/completed/fedcpdp-paper-lopo-20261110/protocol.json), [Chinese execution notes](experiments/completed/fedcpdp-paper-lopo-20261110/README.txt), and [verified results](results/fedcpdp-paper-lopo-20261110/verified_results.json).

## Interpretation and limitations

This is an author-implementation experiment using the paper's reported training budget, **not an exact numerical reproduction**. Dataset counts differ from the paper; the validation proportion and leakage controls are declared implementation choices. The upstream approximate focal-loss gradient and partial alignment backpropagation remain unchanged. Upstream clipping/noise defaults do not establish a verified total differential privacy budget. No method superiority claim is made from this single-method run.

Do not compare scores from different target-label access conditions as though they used identical supervision. Feature-space heterogeneity differs from statistical non-IID data and from model architecture heterogeneity.

## Attribution and license

Original FedCPDP: Meetesh Nevendra and Rahul Shrivastava, *Federated and Privacy-Preserving Cross-Project Defect Prediction With Heterogeneous Features*, DOI [10.1109/TSE.2026.3692541](https://doi.org/10.1109/TSE.2026.3692541). Upstream [MeeteshNevendra/FedCPDP](https://github.com/MeeteshNevendra/FedCPDP), pinned commit `ab5af637a60aed4c34a5338709b3ac65b344ec37`.

Repository-authored code is MIT licensed. Vendored FedCPDP retains its original MIT license and copyright. Third-party datasets and papers are not covered by this repository's license. See [third-party notices](THIRD_PARTY_NOTICES.md).
