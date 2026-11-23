# FedCPDP 隐私—效用权衡结果

**这是独立的隐私扰动—效用研究，不是同构方法对比结果。**

实验 ID：`fedcpdp-native-20261120`；类型：`privacy_utility_tradeoff`。2026-11-23 完成并校验，六条件各 300 次，共 **1,800/1,800** 次、每次 50 轮。底层采用五家族 30 项目的冻结同构 LOPO 数据场景；不与 `results/*homogeneous*` 的历史模型混合。

## 主要发现

- 无扰动：项目宏平均 AUC 0.6652、MCC 0.2145。
- σ=0.1：AUC 0.6896、MCC 0.2253，是当前六条件中平均分最高的一档；相对无扰动的提升在探索性 Holm 校正检验中不显著。
- σ≥0.25：AUC 与 MCC 明显下降。σ=1.0 时 AUC 0.5144、MCC 0.0290。
- 家族效果不一致：σ=0.1 下 AEEEM/NASA/PROMISE 的平均 AUC 上升，而 Relink/SOFTLAB 下降。
- 高噪声下存在 exp 溢出警告和概率饱和，不能把全部性能下降解释为理想噪声机制的纯理论代价。

先对项目的 10 个种子取均值，再对 30 个项目等权汇总。工作量指标只覆盖 LOC 有效的 13 个项目，其余 17 个为 N/A。没有将 300 次重复视为 300 个独立项目。

本实验没有端到端 ε/δ 会计或攻击实测；σ 不是 ε，不能将本结果表述为已经证明真实隐私泄漏随 σ 降低。完整解释见 [Word 实验报告](report/FedCPDP隐私扰动与预测效用实验报告.docx)、[分析数据](report/analysis.json) 和 [效用曲线](report/utility.png)。

## 文件说明

- `conditions/<privacy_condition>/runs/<target>/<seed>/`：原始指标、目标及源验证预测、特征审计、最终进度、训练日志；保留执行字节。
- 每条件的 `all_runs.csv`、`project_summary.csv`、`family_summary.csv`、`metrics_records.json` 与 `verification.json`：完整三种阈值口径及服务器模型重载核验。
- `combined_all_runs.csv`：5,400 行（1,800 个模型 × 三种阈值口径），显式标记 `experiment_type`、`experiment_id`、`privacy_condition` 和 `threshold_mode`。行数不是模型数。
- `report/`：实验报告、科学图表和重算统计，含全部 30 个项目的 AUC/MCC 附录。
- `download_verification.json`：本地归档校验记录；`local_artifact_sha256.json`：包含留在本地的 1,800 个模型检查点在内的原始产物哈希。
- `publication_manifest.json`：本次公开代码与结果的 SHA-256 清单；原始数据、pickle 模型、服务器运维配置、凭据和临时文件不在 Git 中。

原执行 `method` 字段的 `homogeneous` 仅是继承标识；本批次实验归类以目录与显式 `experiment_type` 为准。不得据此把原字段改写后的文件称为执行原件，也不得合并到已有同构主结果。

[运行与复现说明](../../experiments/privacy/fedcpdp-native-20261120/PUBLICATION.md)。在仓库根目录执行 `python scripts/verify_privacy_publication.py` 可核验公开结果，不加载 pickle、不重新训练。

复核预测时使用 pandas `float_precision='round_trip'` 或等价的精确浮点往返解析。默认 CSV 解析可能把接近 1 的不同预测合并为并列分数，导致高噪声条件的 AUC 重算出现小幅差异；不能因此修改原始保存结果。
