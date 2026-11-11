# Dataset provenance

The completed experiment uses the 30 CSV files inside the author's `Dataset.zip`, pinned to commit `ab5af637a60aed4c34a5338709b3ac65b344ec37` of `MeeteshNevendra/FedCPDP`.

Archive SHA256: `4c19cbcf4a8d22fa99f47ec440b99dc716e6148a962d5af4f6f65292137b7801`.

Each CSV hash, project family, label count and raw LOC field is listed in the completed experiment's `manifest.json`. Use `scripts/prepare_fedcpdp_data.py` to validate and prepare a locally obtained archive, or explicitly download the pinned archive. Raw data stays ignored by Git. The original release's count differences from the paper are preserved, not repaired to match reported scores.

Future experiments must record release, project identity/version, schema, label mapping, LOC semantics, hashes and split IDs. Do not count a NASA project again merely because it also appears in a PROMISE repository. Do not fit preprocessing on validation or test labels.
