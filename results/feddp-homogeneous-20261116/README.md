# FEDDP 同构实验结果

2026-11-16 完成并通过核验：**23 个目标 × 10 个种子 = 230 个正式任务，每次 50 轮，失败 0**。PROMISE 11、SOFTLAB 4、AEEEM 4、NASA 4个目标。ReLink只有3个项目，预留公开项目和目标后不足两个私人客户端，因此不适用；所有排除原因见 excluded.json。

这是保留完整方法机制、明确注明差异的重实现，不是未经修改的作者代码或论文表格数值的严格复现。

## 数据权限与解释

独立公开项目用于无标签蒸馏，目标全部测试。公开项目从私人源客户端和目标名单中排除。 各家族分别留一项目，源训练/验证继续使用冻结v3行号；不新增跨客户端重复分组。固定划分下的10个种子只衡量训练随机性，不代表跨划分置信区间。

两种资源条件及目标测试记录不同；本页不与FLR、ALMITY、FedCPDP或另一资源条件作无条件总排名。FedDP的公共项目保留还改变了私人源项目名单，不能仅按相同目标名称视为同条件。

## 主结果

先对每个项目的10个种子取均值，再对项目等权汇总：ROC_AUC 0.7703，MCC 0.3056，Recall 0.5267，G_mean 0.6065，F1 0.4234，CostEffect@20 0.3123，Popt@20 0.3304。

主口径为最后轮模型、源验证客户端宏MCC选阈值；并保留固定0.5阈值结果。工作量指标沿用统一LOC预算定义，不是运行时间；原项目有无效LOC则记为N/A，宏均值只使用有效项目。

FedDP保留LR、随机过采样、LHA相关性权重、按过采样后样本数聚合、加权教师logits和温度KL蒸馏。最后10轮固定0.5阈值的指标均值单列为补充，不替代主结果。

## 文件和核验

- `all_runs.csv`：每次正式任务的指标；`project_summary.csv`、`family_summary.csv`：项目和家族统计。
- `runs/<target>/<seed>/`：指标、目标与源验证预测、数据行号/预处理审计、完整逐轮训练记录。另含最后10轮预测与补充指标；每轮公开采样行号和聚合样本数可审计。
- `verification.json`：预测重算核验记录；`model_verification.json`：全部保存模型重新加载后与预测一致的核验记录。
- `publication_manifest.json`：发布文件哈希；`local_artifact_sha256.json`：包含未发布检查点/日志在内的原始产物哈希。

原始数据、模型二进制、日志和试跑不进入Git；数据通过固定上游提交和30个CSV哈希重建。核验已发布结果不需要下载数据或安装PyTorch：

```bash
python scripts/verify_resource_publication.py --method feddp
```

[代码、环境安装与复现步骤](../../experiments/homogeneous/feddp-resource-20261116/PUBLICATION.md)。
