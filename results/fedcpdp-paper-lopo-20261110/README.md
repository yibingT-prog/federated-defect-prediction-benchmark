# Verified results

`verified_results.json` contains all project-level seed and ensemble metrics from the completed run. The seed list order is 42, 43, 44. `study_mcc` is the study threshold rule; `paper_f1` is the supplementary paper-style rule. `effort` reports all four budgets. JSON null with an accompanying reason denotes an undefined metric, not zero performance.

The source experiment and matching protocol are in `../../experiments/completed/fedcpdp-paper-lopo-20261110/`. The `protocol` basename inside the historical result JSON refers to that experiment's `protocol.json`.

All 30 targets and 90 base models completed. Verification checked labels, row IDs, score finiteness, raw LOC, metrics, model rounds, fingerprints and probability averaging. Ensemble effort metrics were also verified. Per-instance predictions, raw data and model checkpoints remain local and are not published here. Rerun the frozen code to produce these artifacts.
