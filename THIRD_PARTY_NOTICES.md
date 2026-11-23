# Third-party notices

`experiments/completed/fedcpdp-paper-lopo-20261110/vendor/fed_cpdp.py` is an unmodified snapshot from [MeeteshNevendra/FedCPDP](https://github.com/MeeteshNevendra/FedCPDP), commit `ab5af637a60aed4c34a5338709b3ac65b344ec37`.

Copyright (c) 2025 MeeteshNevendra. The full MIT license is preserved at `vendor/LICENSE`. This project is an independent empirical study and does not claim authorship of FedCPDP.

Data manifests describe files used in the study; they do not grant rights to redistribute the underlying datasets. The code repository's MIT license must not be interpreted as a license for NASA, SOFTLAB, AEEEM, Relink, PROMISE, or paper PDFs. Obtain data from the cited upstream source under applicable terms.

The FLR experiment references Yamamoto et al., SANER 2023, and https://github.com/posl/SANER2023_CPDPwithFL. Consulted file identities are recorded in the FLR `author_provenance.json`. Author source files are not redistributed here. TensorFlow, TensorFlow Federated, JAX and other dependencies retain their own licenses; package binaries, Docker layers, raw records and model checkpoints are excluded from this source release.

The ECFML experiment references *A privacy-preserving federated meta-learning framework for cross-project defect prediction in software systems*, Scientific Reports (2025), DOI [10.1038/s41598-025-24440-7](https://doi.org/10.1038/s41598-025-24440-7). The `ecfml_vendor` directory contains the user's local paper-based reimplementation, not author-provided source. Paper text and figures are not redistributed. The local model widths, clipping bound, update budget interpretation and target adaptation settings are disclosed study choices.

The FedDP experiment references *Better Knowledge Enhancement for Privacy-Preserving Cross-Project Defect Prediction* and the paper-linked [waiwaiwang/FedDP repository](https://github.com/waiwaiwang/FedDP). This release contains a portable reimplementation guided by the consulted code. The local consulted file hashes are retained under `provenance`; they are not represented as a verified upstream commit. Unused third-party reference source files are not redistributed. The density-weighted teacher, distillation, oversampling and post-oversampling aggregation counts are retained, with documented runtime fixes and protocol adaptations.

Both 2026-11-16 experiments retrieve the same pinned dataset archive described in `datasets/README.md`. Published predictions, row indices and statistics do not grant a new license to the underlying data. PyTorch and other dependencies retain their own licenses.

The separate `experiments/privacy/fedcpdp-native-20261120/` privacy–utility study reuses the same pinned FedCPDP core. Each condition preserves the upstream MIT license in `vendor/LICENSE`. Raw datasets and model checkpoints are excluded. Report text and figures describe this study's measured results; paper PDFs are not redistributed.
