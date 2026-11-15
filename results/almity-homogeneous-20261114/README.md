# ALMITY 线性 SVM 重实现：同构实验结果

**ALMITY-linear-SVM-paper-reimplementation，不是作者原始复现包。** Ubuntu 服务器完成 30 项目 × 10 种子 = 300 项实验，每项 100 轮，失败 0；2026-11-14 已通过全部模型、预测和指标复核。

## 总体结果

先平均每个项目的10个训练种子，再按项目等权汇总。固定数据划分，种子只改变训练随机性。种子标准差不是跨划分稳健性或置信区间。

- **study**：ROC_AUC 0.6630，MCC 0.1908，Recall 0.4297，G_mean 0.4407，F1 0.3275。
- **fixed_probability**：ROC_AUC 0.6630，MCC 0.1577，Recall 0.1162，G_mean 0.2617，F1 0.1669。
- **native_margin**：ROC_AUC 0.7147，MCC 0.1129，Recall 0.0641，G_mean 0.1784，F1 0.1010。

study 是校准概率经源验证宏 MCC 选阈值的预定主口径；fixed_probability 是校准概率阈值0.5；native_margin 是原生 SVM margin 阈值0。900行记录是300个模型×3种评价条件。
校准斜率为负的任务数：20/300，分别为 ar3 与 ar5 的全部10个种子。校准概率AUC与原生margin AUC分别报告，不因测试集表现切换主口径。

## 家族汇总（study）

- **AEEEM（5项目）**：ROC_AUC 0.7420，MCC 0.3075，Recall 0.5161，G_mean 0.6333。
- **NASA（5项目）**：ROC_AUC 0.5934，MCC 0.0607，Recall 0.4886，G_mean 0.3023。
- **PROMISE（12项目）**：ROC_AUC 0.7224，MCC 0.2321，Recall 0.4435，G_mean 0.5318。
- **Relink（3项目）**：ROC_AUC 0.7508，MCC 0.2266，Recall 0.6484，G_mean 0.3722。
- **SOFTLAB（5项目）**：ROC_AUC 0.4585，MCC 0.0837，Recall 0.1200，G_mean 0.2092。

## 工作量与解释

仅13个项目全部LOC有效，项目等权 CostEffect@20=0.3358，Popt@20=0.3330；其余17个记缺失。使用校准概率/LOC排序和线性插值，表示20% LOC预算下发现的二分类缺陷实体比例，不是bug数量，也不是运行效率。all_runs.csv中native_margin行附带的工作量列仍是这一校准概率排序结果，不应解释成margin排序。

## 方法与协议边界

保留论文Eq.2–6聚合设计；训练集的规模、类别熵、全局少数类占比用于客户端加权。原作者复现包准备时不可访问，SVM核、优化器和完整默认参数未核实。线性hinge+L2、SGD学习率0.01、L2=0.0001、100轮×5本地epoch、batch32、全客户端参与均为本研究预定设置，不能标成论文默认。
五家族内分别留一项目；同一批固定30个CSV，源约80/20、目标全部测试、无公开蒸馏池。冻结行号与FLR以及本机原生FedCPDP一致。保留客户端内部重复分组和源—目标同特征排除，不要求跨客户端分组互斥。源MinMax拟合训练部分，目标自身无标签MinMax属于传导式访问。
sigmoid校准在合并源验证margin上拟合，再复用同一验证集选阈值，这是研究评价适配器，不是核实过的原版ALMITY组件；可能增加选择方差。原生margin结果独立保留。没有正式差分隐私预算保证。
本结果与其他方法的模型、预处理、训练预算不同；此处不作显著优劣排名。旧20261113全局分组协议与本版不混用。

## 发布文件

指标明细、三种口径汇总、逐样本目标/验证预测及aggregation.json均已发布。模型checkpoint.npz、日志、原始/预处理数据保留本地，local_artifact_sha256.json记录原产物哈希；原始数据通过锁定作者提交下载后重建并校验。

[代码与复现步骤](../../experiments/homogeneous/almity-svm-20261114/PUBLICATION.md)。
