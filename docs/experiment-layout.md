# Experiment organization

Use one versioned directory per frozen experiment. A directory contains its runnable entry point, exact protocol, data manifest, provenance, and method dependencies. Completed snapshots are immutable: a change to data, target access, model mechanism, threshold selection or training budget requires a new experiment identifier. Generic method implementations may be shared in future, but their exact revisions must remain recorded.

The `experiments/completed/` area records executed evidence rather than assigning it retrospectively to a research question. In particular, the current FedCPDP run is all-other-project LOPO, not cross-family-only RQ2 and not a sensitivity study.

The four research-question directories reserve the intended organization. Their README files distinguish intended scope from executed implementations. Do not add fake result rows or runnable placeholders to suggest unfinished methods exist.

Every future run should record: method and paper, upstream revision/license, native classifier, feature schema, target label access, public data use, data release and hashes, split IDs, preprocessing fit scope, training and noise seeds, protocol checksum, environment, individual predictions, LOC, metrics, elapsed time and failure reasons.

Store reviewed summaries, numeric target/validation predictions, row-index audits and training histories under `results/<experiment-id>/`, as included in the explicitly requested completed-experiment releases. Keep raw dataset files, model binaries, logs and temporary runs outside Git; record their hashes when needed. Published version-only environment manifests support reproduction; credentials, SSH keys and secret-bearing `.env` files must never be published.

Different methods are complete systems; rankings are not causal estimates of aggregation algorithms. Separate unmatched target-label or resource conditions. Preserve independent repeats separately from ensembles and noise repeats.

The 2026-11-16 releases use separate `ecfml-resource-20261116` and `feddp-resource-20261116` code directories and separate result directories. ECFML has labeled target support/query adaptation; FedDP has independent public-project distillation and only 23 eligible targets. Their full mechanisms, exclusions, implementation assumptions and executed source hashes are documented locally to each experiment. These releases do not change the earlier FLR, ALMITY or FedCPDP evidence.
