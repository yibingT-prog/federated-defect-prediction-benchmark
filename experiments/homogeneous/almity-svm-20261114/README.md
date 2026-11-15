# ALMITY 线性 SVM 重实现：同构实验服务器运行包

交付日期：2026-11-14。协议：rq1-project-lopo-local-groups-v3。

本包替代 20261113 的全局分组方案用于接下来的实验；旧包及旧结果保留。用户已确认保留跨项目预测，仅取消跨客户端联合分组。

**方法名必须写作 ALMITY-linear-SVM-paper-reimplementation（ALMITY 线性 SVM 重实现）。本包不是作者复现包，不声称复现论文表格数值。**

## 1. 方法依据与自主设置

依据 Yang 等人的 TSE 2024 论文 *Federated Learning for Software Engineering: A Case Study of Code Clone Detection and Defect Prediction*。
[论文](https://zpgao.github.io/papers/Federated_Learning_for_Software_Engineering.pdf)；[原复现包地址](https://anonymous.4open.science/r/ALMITY-20B5)。准备时无法取得原复现包，不能核实原 SVM 的核函数、优化器及全部训练参数。

实现论文 Eq.2–6：客户端模型按归一化的 beta 聚合；beta = 数据量占比 Phi × 归一化类别熵 alpha × 全局少数类样本占比 Psi。全局少数类由本任务所有源训练集的类别计数决定；验证集、目标标签不参与权重计算。二分类数量并列时固定选择标签 0，属于实现约定。这里只交换模型和类别计数，没有差分隐私保证。

经用户确认采用线性软间隔 SVM：平均 hinge loss + L2 正则，权重和偏置零初始化。**以下全部是我们的预定研究设置，不是已核实的论文默认值**：100 次实际聚合，每轮本地 5 epoch，batch 32，SGD 无动量，学习率 0.01，L2 系数 0.0001，全部客户端参与，不重采样。目标训练种子 0～9。每次均取第 100 轮模型，不看目标结果挑选轮次。

代码用 NumPy 在 CPU 上计算，不依赖 PyTorch、TensorFlow、CUDA。不能将它称作原版 ALMITY 的严格复现，或将结果差异完全归因于聚合算法。

## 2. 数据、项目隔离与本次变更

data/ 包含之前 FedCPDP/FLR 使用的固定 30 个 CSV，共 15,901 行。来源为 [FedCPDP 仓库](https://github.com/MeeteshNevendra/FedCPDP)，此前固定提交 ab5af637a60aed4c34a5338709b3ac65b344ec37；文件 SHA256 见 data_manifest.json。不声称这批版本是各项目最新版本。本包用于服务器实验交接，不授予第三方数据再发布许可。

五个家族各自留一项目测试：

- PROMISE：ant-1.7、camel-1.6、ivy-2.0、jedit-4.0、log4j-1.0、lucene-2.4、poi-3.0、synapse-1.2、tomcat、velocity-1.6、xalan-2.4、xerces-1.3。
- SOFTLAB：ar1、ar3、ar4、ar5、ar6。
- AEEEM：EQ、JDT、LC、ML、PDE。
- NASA：CM1、MW1、PC1、PC3、PC4。
- Relink：Apache、Safe、Zxing。

一个项目对应一个源客户端。每个任务分别有 11、4、4、4 或 2 个源客户端，目标项目全部原始记录用于最终测试。30 个目标 × 10 个训练种子 = 300 次训练。例如目标 ar5 的源只有 ar1、ar3、ar4、ar6。不存在把目标项目 80% 数据拿来训练的操作。

**论文依据与我们的研究设置分别说明：**

- 项目级隔离参考 FLR 原论文 PDF 第 5 页及 FedDP 第 11 页：目标项目及其其他版本不能进入训练。这两篇原论文还使用其他项目的多个源版本；本包只保留固定快照中的每项目一个版本，不能称为完整照搬原论文数据设置。
- 源端约 80% 训练、20% 验证，用验证集选择阈值，是我们的统一评价设置。FLR 原论文固定阈值 0.5，没有本包这套验证选阈值流程。
- ALMITY 原论文的客户端内部 80%/10%/10% 不在本包采用；本包评价其重实现方法在跨项目预测条件下的表现。

**本次只取消跨客户端联合分组：**

1. 原有源—目标相同特征排除规则保留：源中与目标完整特征向量相同的记录排除；目标原记录全部保留，排除判断不读取目标标签。
2. 客户端内部仍将相同完整特征向量作为一组，以避免同客户端训练/验证两侧出现同组记录。每个源客户端独立按标签进行五折分层分组，固定种子 20261110，选择两侧都含两类的首个合法折作为验证集，其余作为训练集。
3. **不同客户端独立划分，不统一分配重复组、不因跨客户端特征重叠拒绝任务。**因此不同客户端的训练和验证中可能出现相同特征向量。本协议不声称跨客户端特征全局互斥。
4. 实际验证占比为 **12.96%～22.56%**，并非严格 20%。198 个客户端划分的数量和折号见 split_audit.json，原始零起始行号见 prepared/*_splits.json。
5. 本包重新生成的全部行号已与完成的 FLR 同构实验逐项核对，一致；原始参考行号一并封存在 split_reference/，服务器重建不依赖本地旧实验目录。flr_alignment.json 记录核验结果。
6. 全部模型种子 0～9 共用固定划分；种子只改变训练随机性。每个源客户端的 MinMax 只拟合自身训练数据，再应用于自身验证数据；目标使用自身无标签特征的 MinMax，属于允许目标无标签访问的传导式条件。
7. 不预留公开蒸馏池或攻击池，不重采样。camel 和 ar1 正常轮流作为目标或源。

原始数据、源项目名单、训练/验证行号、目标记录和归一化规则均与既有 FLR 对齐；ALMITY 使用 float64，FLR 的变换后特征存为 float32，强制转为相同类型后逐项一致。现有 FLR 结果可在用户选定的数据协议下使用，不必仅为此次取消联合分组重跑。各方法模型、训练预算和 SVM 校准适配差异仍须独立说明。

其他方法应沿用本包同一套行号和目标访问条件。20261113 包的全局分组 v2 结果不能与本版混用。此次数据检查确认字段一致及上述划分规则，不代表重新考证了所有原始指标的历史语义与单位。

## 3. 评价适配与结果解释

SVM 原生输出是 margin。原生结果完整保留在 native_margin：margin >= 0 分类、连续 margin 计算 ROC-AUC。

为了沿用此前研究中概率阈值网格和概率/LOC 排序，另加**研究用评价适配器，不属于核实过的原版 ALMITY 组件**：在合并的源验证集上，以 margin 拟合一维 sigmoid 逻辑回归校准（C=1，lbfgs）；再用同一批源验证集选择概率阈值。校准不使用目标标签，也不改变 SVM 的训练。校准斜率及截距随结果保存；若斜率为负，校准后的 AUC 与原生 margin AUC 可不同，应同时检查，不将两者混称原生结果。复用验证集拟合校准和阈值可能增加选择方差，已明确记录。

- study：概率阈值从 0.05～0.95、步长 0.025 中选，使源客户端宏平均 MCC 最大；并列依次选宏 Recall 更高、阈值更低。
- fixed_probability：校准概率固定阈值 0.5。
- native_margin：原生零间隔阈值，独立记录，不与前两种混算。
- 分类指标：ROC-AUC、MCC、Recall、G-mean，附 F1。
- 工作量指标：概率/原始 LOC 降序，原始行号打破并列；预算内分段线性插值，报告 CostEffect、截断 Popt 的 5/10/20/40%，主报告 20%。只有 13 个项目所有 LOC 有效；其余 17 个保留分类结果、工作量指标记 null/N/A。不改 LOC，不删掉无效项目。
- 工作量指标属于校准概率排序，不能解释为原生 SVM margin 的工作量指标。它不是运行耗时评价。
- 汇总先对同一项目的 10 个种子取均值，再对项目取均值；不把 300 次训练视为 300 个独立项目。缺种子时汇总会保留 count，verification.json 标为 partial。

## 4. 建议服务器

本版本不需要 GPU。专门运行它可选 **4 vCPU、16 GB 内存、50 GB SSD、Ubuntu 24.04 LTS x86_64**，启动 2～4 个进程；后续同时运行其他模型可选 8 vCPU、32 GB，先用 4 进程。

可选阿里云 ECS 通用型按量付费；[官方规格](https://help.aliyun.com/zh/ecs/user-guide/general-purpose-instance-families/)中 ecs.g8i.2xlarge 为 8 vCPU/32 GiB，区域是否有货和实际价格以购买页为准。不建议为此线性 SVM 单独租 A100/4090。AutoDL 的[官方无卡模式](https://www.autodl.com/docs/save_money/)只有 0.5 核/2 GB，适合上传调试，不按其 GPU 机的完整 CPU 配置估算。

本地已能完成运行验证，因此租服务器是为独立执行和批量管理，不是算法必需。这里不承诺全量完成时间，亦不开展算法运行效率比较。

## 5. 上传、安装、检查、启动

把 zip 上传服务器并解压，进入解压后的 almity-svm-server-20261114 目录。使用 Python 3.11～3.13；Ubuntu 24.04 默认 Python 3.12 可用。依赖已固定。下面以 Ubuntu 为例：

```bash
sudo apt-get update
sudo apt-get install -y python3-venv unzip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_package.py
python check.py
python train.py --target ar5 --seed 0 --smoke
nohup python -u schedule.py --workers 4 > scheduler.log 2>&1 &
```

不必运行 prepare.py；包内已包含冻结的 prepared 数据。若主动重建，必须使用固定依赖后执行 check.py，并比较 package_manifest.json；不要在训练过程中重建数据或修改配置。

正式启动命令默认跑全部 30×10。运行包不携带本地 runs/，服务器从头训练；sample_results/ 仅作输出示例，不会被调度器识别为已完成任务。

分批运行可用 `python schedule.py --workers 2 --targets ar5 ar6 --seeds 0 1`；这只完成所选 4 项，不能称全部完成。整个目录同时只运行一个调度器，scheduler.lock 防止重复启动；不要另起直接 train.py 写入同一个任务目录。

查看进度：

```bash
cat status.json
tail -n 30 scheduler.log
cat runs/ar5/0/progress.json
tail -n 20 runs/ar5/0/console.log
```

status.json 保存活动任务、完成名单、失败名单和待运行数；每个任务 progress.json 保存已完成通信轮数。任务的 console.log 每 10 轮记录一次训练进度。后台运行后断开 SSH 不会主动停止调度。

## 6. 暂停、恢复、异常处理

在运行目录执行 `touch STOP`，等待各任务完成当前轮、调度器退出，再关机。确认状态后执行 `rm STOP`，用同一启动命令恢复。完成任务按数据/代码指纹跳过，未完成任务从最近完整轮恢复；本地已验证中断后结果一致。

如果机器被强制关机，scheduler.lock 可能残留。先用 `ps -ef | grep '[s]chedule.py'` 和 `ps -ef | grep '[t]rain.py'` 确认没有本目录对应进程，才移除本目录的锁文件并重启。不能在现有进程还写入时删除锁并再次调度。

出现 fingerprint 不一致，应保留原结果并在一个新目录开始新实验，不手工修改指纹绕过检查。失败任务的 console.log 会保留原因，修复环境后同命令重新调度。不要通过更改方法参数来继续同一个实验目录。

## 7. 输出、核验和交回结果

训练完成后执行 `python summarize.py`；只有 verification.json 的 verified_runs=300 且 status=complete，并且 status.json 没有失败任务，才能报告全量完成。

- runs/<目标>/<种子>/：checkpoint.npz、predictions.csv、metrics.json、validation_*.csv、aggregation.json、progress.json、console.log。
- all_runs.csv：每次训练的三种评价条件分开保存，共 900 行才是完整矩阵。
- project_summary.csv：逐项目、逐条件的均值、标准差和数量。
- family_summary.csv：先平均项目内种子，再平均家族内项目。
- verification.json：根据保存模型、目标数据、预测和验证记录重新计算指标的核验状态。

交回完整 runs/、汇总文件、status.json、scheduler.log、protocol.json、tasks.json、data_manifest.json、package_manifest.json 和 `python -m pip freeze` 输出，不只交一个平均数。已有 GitHub 仓库中的 FedCPDP、FLR 结果目录与此包分开，本包尚未上传 GitHub。

## 8. 本地验收范围

30 个目标均完成 2 轮短程检查；ar5 的种子 0、1 均完成完整 100 轮。种子 0 的独立连续训练与第 2 轮后恢复训练，最终权重逐位相同、预测文件逐字节相同。还检查了聚合手算值、hinge 梯度有限差分、指标基本性质、30 套数据的项目隔离、客户端内部隔离及归一化重构（不要求跨客户端特征互斥）。scheduler 已验证小批量调度与已完成任务跳过。

验证在 Windows/Python 3.13.5 完成；**尚未在真实 Linux 云服务器上运行验证**。源码使用相对路径和跨平台 Python 接口，服务器先按第 5 节进行检查。preflight_summary.json 和 local_validation.json 记录事实；sample_results/ 是少量本地验证输出，不是全量实验结果。preflight.py 可重做验收，但不是运行正式实验的必要步骤。

单项目验证结果不用于证明方法优劣或调整参数。完整实验之后再与同一本版数据协议下其他模型做配对比较。
