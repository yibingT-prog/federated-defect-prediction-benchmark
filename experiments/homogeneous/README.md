# RQ1: homogeneous feature spaces

[FLR TFF experiment](flr-tff-20261111/README.md) completed 300 runs on 30 within-family LOPO targets. Its post-run audit disclosed cross-client training/validation duplicate overlap in 21 targets; see the experiment README before interpreting results. A complete matched multi-method RQ1 comparison remains pending. A shared dataset family alone does not prove feature compatibility.

[Completed FedCPDP native homogeneous experiment](fedcpdp-native-20261114/PUBLICATION.md): 300 models, 50 rounds each, original author training-loss aggregation. [Results](../../results/fedcpdp-native-homogeneous-20261114/README.md). Source row splits match the frozen FLR/ALMITY protocol; native preprocessing and budgets differ. The chosen protocol does not isolate duplicates jointly across source clients.
