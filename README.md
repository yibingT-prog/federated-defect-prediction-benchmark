# Federated Defect Prediction Benchmark

Reproducible experiments comparing complete federated learning methods for software defect prediction. Native classifiers and core method mechanisms are preserved. The repository is intended to cover homogeneous features, heterogeneous features, privacy effects, and sensitivity analysis.

**Current release:** seven separately versioned completed experiments:

- [FedCPDP mixed-source LOPO](experiments/completed/fedcpdp-paper-lopo-20261110/): historical 90 seed models and 30 ensembles.
- [FLR homogeneous](experiments/homogeneous/flr-tff-20261111/): 300 models; [results](results/flr-homogeneous-20261111/README.md).
- [FedCPDP native homogeneous](experiments/homogeneous/fedcpdp-native-20261114/PUBLICATION.md): 300 models, 50 rounds, author training-loss aggregation; [results](results/fedcpdp-native-homogeneous-20261114/README.md). The earlier server validation-loss variant is separate.
- [ALMITY linear SVM reimplementation](experiments/homogeneous/almity-svm-20261114/PUBLICATION.md): 300 models, 100 rounds; [results](results/almity-homogeneous-20261114/README.md).
- **[ECFML complete-mechanism reimplementation](experiments/homogeneous/ecfml-resource-20261116/PUBLICATION.md):** 300 models across five families, 50 rounds, **labeled target support adaptation**; [results](results/ecfml-homogeneous-20261116/README.md).
- **[FedDP complete-mechanism reimplementation](experiments/homogeneous/feddp-resource-20261116/PUBLICATION.md):** 230 models across four eligible families, 50 rounds, **independent public distillation data**; [results](results/feddp-homogeneous-20261116/README.md). ReLink cannot retain two private clients after reserving public and target projects and is explicitly not applicable.
- **[FedCPDP privacy–utility tradeoff](experiments/privacy/fedcpdp-native-20261120/PUBLICATION.md):** 1,800 new runs, six clipping/noise conditions × 30 targets × 10 seeds; [separate results and report](results/fedcpdp-privacy-utility-20261120/README.md). Completed 2026-11-23. This is a privacy–utility study, **not a homogeneous method-comparison experiment**, even though it uses frozen within-family homogeneous data. No verified end-to-end epsilon/delta claim.

ECFML and FedDP completed and passed prediction/model verification on 2026-11-16. These are declared reimplementations, not untouched author programs or exact paper-table replications. Target-label access, public resources, private source lists and test records differ between conditions; their aggregate scores do not form a controlled method ranking. Multi-method rankings and sensitivity studies remain pending.

**FLR audit limitation:** 21 target tasks have cross-client source training/validation duplicate-feature overlap. Numerical results passed verification, but threshold selection can be affected. This historical experiment is published with limitations, not certified as a globally leakage-free final benchmark. See the [FLR results and analysis](results/flr-homogeneous-20261111/README.md). The FedCPDP run includes same-family sources and is not the final cross-family-only RQ2 benchmark.

## Repository layout

```text
experiments/
  completed/fedcpdp-paper-lopo-20261110/  # immutable executed source and protocol
    run.py, check.py, monitor.py
    protocol.json, manifest.json, provenance.json
    vendor/fed_cpdp.py, vendor/LICENSE
  homogeneous/flr-tff-20261111/         # executed native-TFF FLR + frozen splits
  homogeneous/fedcpdp-native-20261114/  # author native training-loss aggregation
  homogeneous/almity-svm-20261114/      # declared linear SVM reimplementation
  homogeneous/ecfml-resource-20261116/  # target labeled support + held-out query
  homogeneous/feddp-resource-20261116/  # independent public-project distillation
  heterogeneous/                       # RQ2 scope and admission requirements
  privacy/fedcpdp-native-20261120/       # RQ3 privacy–utility: six native perturbation conditions
  sensitivity/                         # RQ4 scope, not yet executed
datasets/                              # provenance and local preparation policy
results/fedcpdp-paper-lopo-20261110/     # verified project-level metrics
results/flr-homogeneous-20261111/        # separate FLR metrics, analysis and audit
results/fedcpdp-native-homogeneous-20261114/ # 300 native models, metrics and predictions
results/almity-homogeneous-20261114/     # 300 linear SVM models
results/ecfml-homogeneous-20261116/      # 300 models, separate supervision condition
results/feddp-homogeneous-20261116/      # 230 models, final-round and last-10 results
results/fedcpdp-privacy-utility-20261120/ # 1,800 privacy–utility runs, separate from method rankings
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

## FLR / ALMITY 隐私扩展交付（2026-11-24）

本分支新增两方法的扩展源码、整理后的正式结果与中文实验报告/交接。请从 [交付入口](FLR_ALMITY_PRIVACY_DELIVERY.md) 阅读；每方法 1800 个正式模型。原基线和既有结果保持原样，缺失验收证据及独立模型/数据归档在交接中说明。
