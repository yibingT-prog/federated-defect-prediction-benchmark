# FedCPDP 作者原生训练的同构实验

用户于 2026-11-14 批准本地运行。新目录独立保存，不改动已发布混合家族实验、旧同构预实验或 ALMITY/FLR。

作者代码固定提交 ab5af637a60aed4c34a5338709b3ac65b344ec37；vendor/fed_cpdp.py 保留原文件及许可证，每个任务验证 SHA256。直接调用作者 federated_round、local_train、aggregate，不替换训练损失为验证损失，不重新实现作者梯度。

## 协议

- 使用固定30个项目、五个家族，家族内留一项目测试。每个源项目一个客户端；目标全部测试。
- 源训练/验证行号与 ALMITY 20261114、已完成的 FLR 一致，复制到 splits/。约80/20，种子20261110；客户端内部同特征分组、排除源中目标同特征记录，不执行跨客户端联合分组。
- 模型种子0～9共享划分，共300个模型。无种子集成，主结果第50轮，不按测试表现选轮。
- 保留作者Z-score、哈希1024、MLP1024→128→64→1、ReLU、Dropout0.2、Focal实现、Adam及本地余弦调度、hybrid聚合、服务器动量、隐藏均值方差统计更新。
- 原生特征排除规则保留，每任务features.json记录输入字段和哈希位置。源归一化只拟合训练，验证使用相同统计量；目标自归一化属于无标签目标访问。此项与FLR/ALMITY的MinMax不同。
- 使用论文预算覆盖：50轮、本地5epoch、batch32、FedProx0.01、均值/方差对齐均0.001。学习率0.003、L2为0.0001沿用作者配置。
- 显式开启裁剪加噪，C=1，noise_multiplier=0.1；作者默认USE_DP=False。未核验总epsilon/delta，不宣称完整隐私会计。
- 作者脚本原默认80轮/12epoch/batch128/FedProx0.005/均值对齐0.003。此实验是作者原生机制在声明配置下的同构比较，不是所有默认值原封不动的论文数值复现。
- 无公开蒸馏池、无重采样。固定PYTHONHASHSEED=0，每进程数值库1线程，两个任务并行。

## 评价

最后一轮源验证概率选阈值。主规则为0.05～0.95、步长0.025中源项目宏MCC最大，并列宏Recall最大、阈值最小。补充固定0.5及合并源验证F1最大阈值；原生脚本在训练集集成预测上选阈值，本实验使用独立源验证和10个单模型，是统一评价适配。验证数据不参加聚合损失计算。

ROC-AUC使用连续概率；MCC、Recall、G-mean、F1使用选定阈值。工作量指标按概率/原始LOC排序，预算内线性插值，截断Popt；报告5/10/20/40%，主20%。17个项目存在无效LOC，工作量指标记null，其分类结果保留。先平均项目内10个种子，再项目层面汇总，不以300次训练代替30个项目。

## 运行与结果

PowerShell：`$env:PYTHONHASHSEED='0'` 后运行 `python preflight.py`。通过后运行 `python schedule.py --workers 2`，调度器也会为子进程设置哈希种子。可从同一命令断点恢复；仅允许一个调度器。

status.json：总体任务状态；runs/<目标>/<种子>/progress.json：轮次；console.log：逐轮日志。checkpoint.pkl是本次本地产生的可信检查点，不加载外部未知pickle文件。每轮保存模型、服务器状态与随机状态。

创建STOP空文件后，各任务完成当前轮暂停。恢复前移除STOP；强制中断后的scheduler.lock只有确认本目录无运行进程后才可清除。不要修改运行中代码、数据或协议，否则指纹失配。

结果包括predictions.csv、validation_*.csv、metrics.json、features.json、checkpoint.pkl。summarize.py从检查点重新预测、重算阈值与指标，再生成all_runs.csv、project_summary.csv、family_summary.csv。verification.json为complete且verified_runs=300才算全部核验完成。

验收记录：ar5 的作者 Sigmoid 出现 exp 溢出警告；最终概率有限，原生训练逐位一致与结果核验通过。保留原实现和警告，不根据该结果调参。详见 preflight.json。
