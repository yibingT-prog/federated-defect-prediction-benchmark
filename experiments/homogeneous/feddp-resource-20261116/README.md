# FEDDP homogeneous resource-aware experiment

This is a complete-mechanism, explicitly documented reimplementation under the unified frozen five-family benchmark, not an exact numerical reproduction of the paper.

See protocol.json for declared choices and unavailable author settings; tasks.json and excluded.json are authoritative. Expected 230 runs, seeds 0--9, 50 rounds.

ECFML uses target labeled support (approximately 80%) and evaluates only the held-out query (approximately 20%). All source-project validation records remain outside meta-training. A local support/query sampler prevents the same row from appearing in both minibatches even for small clients. Query labels never select parameters or thresholds. Query-batch BatchNorm statistics are used by the inherited network. This setting cannot be pooled into the zero-target-label primary ranking.

FedDP reserves an entire independent public project within each family. Public labels are never used in training, LHA, or distillation. Its public data are not private clients or targets. ReLink cannot meet the two-private-client minimum after reservation and is marked not applicable. SOFTLAB uses all 121 ar1 records per round because the author's sampler takes min(700,n); no extra records are invented. AEEEM and NASA are extensions beyond the original FedDP families. Compare only matching resource and target conditions.

Source splits reuse v3 exactly, including target-feature duplicate exclusion and per-client grouping, without cross-client duplicate regrouping. Public-source feature duplicates are retained, consistent with the chosen no-global-grouping protocol; their counts are audited. Standardization or MinMax statistics and complete row indices are saved per run. ECFML target support/query groups remain disjoint. Source validation chooses the main threshold; native 0.5 results are also saved. Metrics share the previous implementation. Invalid original LOC makes effort metrics N/A.

Run `python test_mechanisms.py`, then `python verify.py --preflight`, then `python schedule.py --workers 2`. Use `python train.py --target TARGET --seed 0 --rounds 2 --out smoke` for a short check. Full jobs resume at the last atomic checkpoint, including RNG state. Create STOP to stop after active tasks; remove it and restart the scheduler to resume. Never launch two schedulers in this directory. `python verify.py` requires every expected result and recomputes metrics from predictions.

Training is simulated federated training on one machine, with separate logical client datasets. Saved checkpoints, results, and all run logs are local artifacts. No central result is used to claim deployable privacy protection. Wall-clock logs support operations only, not an efficiency comparison.
