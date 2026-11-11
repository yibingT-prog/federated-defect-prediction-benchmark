# Experiment organization

Use one versioned directory per frozen experiment. A directory contains its runnable entry point, exact protocol, data manifest, provenance, and method dependencies. Completed snapshots are immutable: a change to data, target access, model mechanism, threshold selection or training budget requires a new experiment identifier. Generic method implementations may be shared in future, but their exact revisions must remain recorded.

The `experiments/completed/` area records executed evidence rather than assigning it retrospectively to a research question. In particular, the current FedCPDP run is all-other-project LOPO, not cross-family-only RQ2 and not a sensitivity study.

The four research-question directories reserve the intended organization. Their README files distinguish intended scope from executed implementations. Do not add fake result rows or runnable placeholders to suggest unfinished methods exist.

Every future run should record: method and paper, upstream revision/license, native classifier, feature schema, target label access, public data use, data release and hashes, split IDs, preprocessing fit scope, training and noise seeds, protocol checksum, environment, individual predictions, LOC, metrics, elapsed time and failure reasons.

Store small reviewed project-level result summaries under `results/<experiment-id>/`. Keep raw datasets, per-instance outputs, checkpoints and temporary outputs outside Git. A later release of those artifacts requires its own data-rights and disclosure review. Never publish credentials or local environment files.

Different methods are complete systems; rankings are not causal estimates of aggregation algorithms. Separate unmatched target-label or resource conditions. Preserve independent repeats separately from ensembles and noise repeats.
