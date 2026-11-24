"""Build the Chinese experiment report and 15-section handoff from audited results."""
from __future__ import annotations
import collections
import csv
import hashlib
import json
import math
import re
import shutil
import statistics
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / "实验报告与交接材料"
REPORT_NAME = "FLR与ALMITY隐私实验_实验报告.md"
HANDOFF_NAME = "FLR与ALMITY隐私实验_开发运行与迁移交接.md"
CONDITIONS = ("none", "clip_only", "sigma_0p1", "sigma_0p25", "sigma_0p5", "sigma_1p0")
METHODS = ("FLR", "ALMITY")
METRICS = ("ROC_AUC", "MCC", "Recall", "G_mean", "F1")
audit = json.loads((EVIDENCE / "文档依据核对.json").read_text(encoding="utf-8"))


def rows(method):
    p = ROOT / f"experiments/privacy/{method.lower()}-gaussian-extension-v1/runs/formal/run_metrics.csv"
    return [r for r in csv.DictReader(p.open(encoding="utf-8-sig", newline="")) if r["threshold_mode"] == "study"]


STUDY = {method: rows(method) for method in METHODS}


def table(headers, data):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"] + ["| " + " | ".join(str(v) for v in row) + " |" for row in data])


def target_means(method, condition, metric, family=None):
    values = collections.defaultdict(list)
    for r in STUDY[method]:
        if r["privacy_condition"] == condition and (family is None or r["family"] == family) and r[metric] not in (None, ""):
            values[r["target"]].append(float(r[metric]))
    return {t: statistics.mean(v) for t, v in values.items()}


def mean(method, condition, metric, family=None):
    return statistics.mean(target_means(method, condition, metric, family).values())


def performance_table(method):
    data = []
    for c in CONDITIONS:
        cells = [c]
        for metric in METRICS:
            v = list(target_means(method, c, metric).values())
            cells.append(f"{statistics.mean(v):.4f} ± {statistics.stdev(v):.4f}")
        data.append(cells)
    return table(["条件", "ROC-AUC", "MCC", "Recall", "G-mean", "F1"], data)


def delta_table(method):
    return table(["条件", "ΔAUC / none", "ΔAUC / clip_only", "ΔMCC / none", "ΔRecall / none", "ΔG-mean / none"], [[c, f"{mean(method,c,'ROC_AUC')-mean(method,'none','ROC_AUC'):+.6f}", f"{mean(method,c,'ROC_AUC')-mean(method,'clip_only','ROC_AUC'):+.6f}", *[f"{mean(method,c,m)-mean(method,'none',m):+.6f}" for m in ("MCC", "Recall", "G_mean")]] for c in CONDITIONS])


def family_table(method):
    families = ("AEEEM", "NASA", "PROMISE", "Relink", "SOFTLAB")
    return table(["条件", *families], [[c, *[f"{mean(method,c,'ROC_AUC',f):.4f}" for f in families]] for c in CONDITIONS])


def effort_table():
    return table(["方法", "条件", "有效目标数", "CostEffect@20", "Popt@20"], [[m,c,"13/30",*[f"{mean(m,c,k):.4f}" for k in ("CostEffect@20","Popt@20")]] for m in METHODS for c in CONDITIONS])


def diagnostic_table():
    return table(["方法", "条件", "正式任务", "饱和任务", "最高饱和比例", "边界阈值任务", "校准斜率<0任务"], [[m,d["condition"],d["tasks"],d["saturated_tasks"],f"{d['max_saturation_fraction']:.2%}",d["threshold_edges"],audit["methods"][m]["calibration"].get(d["condition"],{}).get("negative_slope_tasks","不适用")] for m in METHODS for d in audit["methods"][m]["diagnostics"]])


def family_inventory():
    fam = audit["methods"]["FLR"]["families"]
    return table(["家族", "目标数", "目标项目（大小写、版本固定）", "每任务客户端数"], [[f,len(fam[f]),"、".join(fam[f]),len(fam[f])-1] for f in ("AEEEM","NASA","PROMISE","Relink","SOFTLAB")])


def archive_table():
    return table(["本地归档（相对本文件）", "字节数", "SHA-256", "用途"], [[f"[{Path(p).name}]({p})",v["bytes"],f"`{v['sha256']}`","完整项目与检查点" if "project-and-results" in p else ("FLR Linux 环境" if "environment.tar" in p else "精简分析/源码包")] for p,v in audit["archives"].items()])


def ensure_audit():
    for method in METHODS:
        info = audit["methods"][method]
        assert info["unique_models"] == 1800 and not info["missing"] and not info["extra"]
        assert not info["invalid_progress"] and not info["summary_recalculation_errors"]
        assert info["prepared_content_hash_passed"] == 30 and info["cross_method_frozen_split_matches"] == 30


def main():
    ensure_audit()
    report = f"""# FLR 与 ALMITY 隐私—效用权衡扩展实验报告

版本：2026-11-24 / v1.0。实验标识：`flr-gaussian-extension-v1`、`almity-gaussian-extension-v1`。

配套文档：[开发、运行与迁移交接]({HANDOFF_NAME})；[原始开发与运行交接要求](隐私-效用权衡扩展实验_开发与运行交接.md)。原始交接文件未修改。本报告基于已迁出的结果与源码，不依赖服务器继续在线。

## 摘要

本实验在冻结的五家族、30 个目标项目上，对 FLR 与 ALMITY 增加聚合前的客户端参数更新整体 L2 裁剪与高斯扰动，考察预测效用随扰动强度的变化。每方法运行六条件、十个模型种子，共 1800 个正式任务，均达到 100 轮。现有模型/预测验收报告均为 passed、错误数 0；本次文档编写又独立核对了任务组合、进度、数据内容哈希和汇总统计。

FLR 项目宏平均 ROC-AUC 从无扰动的 0.7305 变为 sigma_1p0 的 0.7030，变化为 −0.0275；ALMITY 从 0.6630 变为 0.5915，变化为 −0.0716。较小扰动下各指标并非同步单调变化，因此不据此选择所谓最优部署噪声。FLR 有 184 个任务包含极端概率；ALMITY 无极端概率，但部分源验证校准斜率为负，需要与原生 margin 结果分开解释。这些结果只描述固定操作网格下的预测效用，不证明端到端差分隐私水平。

## 1. 实验目的、范围与术语

研究问题是：保留各方法的学习器、训练预算、聚合和评价适配时，整体裁剪及不同高斯扰动强度如何影响方法内部的预测效用。本次仅包含 FLR 与 ALMITY；FedProx、FedDP、FedCPDP 和 ECFML 不计入本次 3600 个模型任务。其他方法可能保存在完整仓库备份中，它们属于历史材料。

| 术语 | 本报告中的固定含义 |
| --- | --- |
| FLR | 原生 TensorFlow Federated（TFF）0.20.0 执行的单层逻辑回归适配版本 |
| ALMITY | 按论文重实现的线性 soft-margin 支持向量机（SVM），非作者原始实现 |
| privacy_condition | 六种客户端更新扰动条件 |
| threshold_mode | 同一模型的评价口径，不代表一个新训练任务 |
| C、sigma | 整体裁剪阈值与高斯噪声乘子，sigma 不是 epsilon |
| 项目宏平均 | 先平均同一目标的十个种子，再对目标等权平均 |

## 2. 数据、划分与访问权限

数据为用户提供的 [Dataset.zip](Dataset.zip)，SHA-256 为 `4c19cbcf4a8d22fa99f47ec440b99dc716e6148a962d5af4f6f65292137b7801`，与原交接要求一致。上游为 FedCPDP 数据发布提交 `ab5af637a60aed4c34a5338709b3ac65b344ec37`；不得另取同名 CSV 替换。

{family_inventory()}

执行家族内留一项目测试，一个源项目对应一个客户端，全客户端参与。源客户端训练/验证行号约为 80/20，沿用冻结索引，模型种子 0–9 不改变划分。原始目标项目全部记录用于最终测试；目标标签不用于本地训练、校准或选阈值。源训练 min-max 参数用于相应源验证；目标特征按目标自身无标签特征做 min-max。常量列分母采用 1。

保留客户端内部特征重复分组，以及源端与目标同特征记录排除；没有新增跨源客户端全局去重。因此不能宣称所有源训练和源验证之间全局无相同特征记录。未增加目标支持集、公开辅助训练集、过采样或蒸馏池。

FLR 准备数组为 float32，ALMITY 为 float64。本次独立核对两方法各 30 份准备数组内容哈希均匹配各自冻结值，30 个目标的跨方法源划分 JSON 均一致。数值精度差异保留，不能由相同 C、sigma 推断相同隐私水平。

依据：[FLR 数据清单](experiments/homogeneous/flr-tff-20261111/data_manifest.json)、[FLR 任务](experiments/homogeneous/flr-tff-20261111/tasks.json)、[FLR 冻结划分](experiments/homogeneous/flr-tff-20261111/splits/)、[ALMITY 任务](experiments/homogeneous/almity-svm-20261114/tasks.json)、[ALMITY 参考划分](experiments/homogeneous/almity-svm-20261114/split_reference/)。

## 3. 方法与扰动实现

### 3.1 六个条件

| 条件 | 整体裁剪 | C | 每坐标噪声标准差 |
| --- | --- | --- | --- |
| none | 关闭 | 不启用 | 0 |
| clip_only | 开启 | 1.0 | 0 |
| sigma_0p1 | 开启 | 1.0 | 0.1 |
| sigma_0p25 | 开启 | 1.0 | 0.25 |
| sigma_0p5 | 开启 | 1.0 | 0.5 |
| sigma_1p0 | 开启 | 1.0 | 1.0 |

本地训练得到 theta_local 后，delta = theta_local − theta_global。权重与偏置一起展平，执行 delta_clip = delta × min(1, C/(||delta||₂+10⁻¹²))；逐坐标加入均值为 0、标准差为 sigma×C 的正态噪声。加噪后允许范数超过 C，不做第二次裁剪。每个客户端独立处理后再使用原聚合权重；不先加权再裁剪，不将噪声另除以客户端数。

### 3.2 FLR

FLR 使用 `build_federated_averaging_process`，非 none 条件以自定义 WeightedAggregationFactory 逐客户端映射扰动，然后交由原 MeanFactory 作样本数加权聚合；none 不安装该扰动工厂。裁剪覆盖 TFF 传入的全部 trainable tensors，包含偏置；不是在 process.next 返回后对服务器模型加噪。保留 TFF 内部原有服务器更新流程，未人为反转更新符号。

采用单层 Dense(1, sigmoid)、零初始化、二元交叉熵和 SGD；100 次实际聚合更新，每轮本地 100 epoch，batch=20，shuffle buffer=500，客户端学习率 0.1，服务器学习率 1.0。源码见 [FLR 训练](experiments/privacy/flr-gaussian-extension-v1/train.py)、[FLR 扰动](experiments/privacy/flr-gaussian-extension-v1/privacy.py)、[扩展协议](experiments/privacy/flr-gaussian-extension-v1/protocol.json)。

### 3.3 ALMITY

ALMITY 保留线性 SVM 的 hinge loss、L2 正则化、SGD 和原自适应聚合。聚合权重来自源训练样本数比例 Phi、归一化类别熵 alpha、全局少数类数量份额 Psi 的乘积，再归一化；不以普通 FedAvg 替换。非 none 条件扰动 `(lw−w, lb−b)`，重建客户端参数后再聚合；none 直接聚合本地模型。

预算为 100 轮、每轮本地 5 epoch、batch=32、学习率 0.01、L2=0.0001、零初始化、无动量。最终 margin 在源验证集上拟合 LogisticRegression（C=1、lbfgs、max_iter=1000）做 Sigmoid 校准；同一验证集又用于阈值选择，这是既有评价适配，不是原论文必备机制。原生 margin 连续分数 AUC 与 margin≥0 的指标继续保存。源码见 [ALMITY 训练](experiments/privacy/almity-gaussian-extension-v1/train.py)、[扰动](experiments/privacy/almity-gaussian-extension-v1/privacy.py)、[扩展协议](experiments/privacy/almity-gaussian-extension-v1/protocol.json)、[基线公开说明](experiments/homogeneous/almity-svm-20261114/PUBLICATION.md)。

## 4. 实验矩阵、环境与评价方法

每方法为 30 目标×10 种子×6 条件=1800 个正式模型，合计 3600 个模型。每个模型均取预定最后一轮，无测试选轮或早停。FLR 保存 study/native 两种评价，共 3600 行；ALMITY 保存 study/fixed_probability/native_margin 三种评价，共 5400 行。不能把评价行数当作模型数。

| 项目 | FLR 实际备份环境 | ALMITY 实际备份环境 |
| --- | --- | --- |
| 执行形式 | Linux Conda 环境 flr-tff020 | Linux .venv-almity |
| Python | 3.8.20 | 3.12.3 |
| 关键库 | TensorFlow 2.8.0、TFF 0.20.0 | scikit-learn 1.7.2 |
| NumPy / pandas | 1.21.6 / 1.3.4 | 2.3.3 / 2.3.3 |
| SciPy | 1.7.3 | 1.16.2 |
| 线程设置依据 | 调度器设置 TF intra/inter 为 1；数据集线程池为 1 | 调度器设置 OMP/OPENBLAS/MKL/NUMEXPR 为 1 |

版本从已迁出的环境记录/安装包元数据核实，不把历史基线 freeze 当作新机器运行证明。本次没有可核实的 Docker 镜像 ID；实际使用 Conda，不能补造容器镜像。CPU 型号、GPU 型号/是否参与、操作系统发行版和完整实际启动环境变量尚缺独立运行快照；不做硬件或效率排名。

主阈值从 0.05–0.95、步长 0.025 的 37 个候选中，按源项目宏平均 MCC 最大、平局宏 Recall 更高、再平局阈值更低选择。目标标签不参与该选择。ROC-AUC 使用连续分数；主报告使用 study 口径的 ROC-AUC、MCC、Recall、G-mean，另列 F1。

所有表格先计算每目标十种子的均值，再对有效目标等权平均；“±”为这些目标均值之间的样本标准差（ddof=1），不是置信区间，也不是 300 个任务的种子标准差。分类指标有效目标为 30。PROMISE 有 12 个项目，因此“项目等权”不等于“五家族等权”。本报告只作描述性比较，未做显著性检验，不写“显著优于”。

## 5. 正式分类结果

### 5.1 FLR：较大噪声伴随效用下降，小噪声变化有限

{performance_table('FLR')}

{delta_table('FLR')}

FLR 的 clip_only 相对 none 的 AUC 变化为 +0.003677；sigma_0p1 相对 clip_only 为 −0.000418。sigma_1p0 相对 none 的 AUC、MCC、Recall 和 G-mean 均下降。这里的小幅上升只是该固定矩阵的观测，不能解释为裁剪必然改善泛化或某档噪声最优。

### 5.2 ALMITY：校准概率的 AUC 随较强噪声下降

{performance_table('ALMITY')}

{delta_table('ALMITY')}

ALMITY 的 none 与 clip_only 四位小数结果相同，但原始 AUC 差为约 2.48×10⁻⁷、MCC 差约 8.24×10⁻⁶，并非逐位等价。sigma_0p1 的 AUC 仅略降，而 MCC 和 G-mean 略升、Recall 和 F1 降低，说明不同指标反映了排序、阈值和校准的不同方面。sigma_0p25 及以上的 AUC、MCC 整体较低；不能把效用下降写成隐私风险已降低。

### 5.3 家族层面的 ROC-AUC

各单元为该家族内目标均值的等权平均；本表不作跨方法隐私预算排名。

FLR：

{family_table('FLR')}

ALMITY：

{family_table('ALMITY')}

## 6. 工作量指标与缺失值

CostEffect@20 和 Popt@20 依原始 LOC 计算，按 score/LOC 排序并作工作量预算插值。它们是补充指标；原交接对是否纳入最终论文尚留待确认，因此本报告保留结果供核对，不以此改变分类主结论。5%、10%、40% 预算结果仍保留在运行级 CSV，不混入 20% 列。

{effort_table()}

每条件仅 13/30 个目标具备本实现要求的有效 LOC：{'、'.join(audit['methods']['FLR']['valid_effort_targets'])}。其余 17 个目标为 {'、'.join(audit['methods']['FLR']['na_effort_targets'])}；根据当前核验函数，存在非正/缺失 LOC 或不满足工作量评价前提时记为 N/A。未删除 LOC=0 的测试行，分类指标仍使用原测试总体。N/A 不填 0，工作量表也不代表全部 30 个目标。

## 7. 数值诊断与校准解释

饱和定义为某条目标概率 p≤10⁻⁶ 或 p≥1−10⁻⁶；“饱和任务”指至少一条目标预测满足该定义，不代表整个任务的预测全部饱和。边界阈值为 study 阈值等于 0.05 或 0.95。

{diagnostic_table()}

FLR 饱和任务合计 184/1800，边界阈值合计 78/1800；sigma_1p0 的最高任务饱和比例为 6.61%。none 本身也有 20 个饱和任务，因此不能把所有饱和都归因于加噪。ALMITY 无饱和任务、边界阈值共 64 个，但校准斜率为负的任务由 none 的 20/300 增至 sigma_1p0 的 92/300。

负校准斜率意味着概率随原生 margin 反向变化，并非程序必然报错；它会改变概率排序方向，因此 ALMITY 校准概率 AUC 不应与原生 margin AUC混算。应同时查看 `native_margin`，不能以取消高噪声校准的方式选择性修饰结果。本次未汇总出一份覆盖全部日志的警告类型清单，故“error_count=0”不等于“从未发生警告”。

## 8. 验收证据、独立复核与证据缺口

| 核对项目 | FLR | ALMITY | 证据范围 |
| --- | --- | --- | --- |
| 正式模型组合 | 1800/1800 | 1800/1800 | 本次独立核对 condition/target/seed，无缺失/重复 |
| 正式完成轮次 | 全部 100 | 全部 100 | 本次检查逐任务 progress 与 metrics |
| 保存验收结果 | passed，错误 0，3600 行 | passed，错误 0，5400 行 | 历史 verification.json；检查点重载、预测与指标核验 |
| 准备数组内容哈希 | 30/30 匹配 | 30/30 匹配 | 本次独立按键名、dtype、shape、内容重算 |
| 跨方法冻结划分 | 30/30 一致 | 30/30 一致 | 本次比较 splits 与 split_reference |
| 方法均值与标准差 | 全部匹配 | 全部匹配 | 本次从运行级 study 表独立重算 |
| 简单机制单测 | 现有脚本可验证 | 现有脚本可验证 | [3,4] 裁剪、none、非有限输入；不是全部原要求测试 |
| 原基线/none 同环境参数对齐与服务器等价性 | 未见足够的独立报告 | 未见足够的独立报告 | 不因最终预测可重算就视作已证明 |
| 两目标完整轮次试运行 | 有 12 个完成标记；不保存试运行预测 | 完整备份有 12 个任务 | FLR 尚不能据完成标记认定试运行全套预测验收完成 |
| 中断恢复连续性、调度锁与独立噪声流 | 尚有待补项 | 尚有待补项 | 具体见交接第 10–11 节 |

重要边界：现有 verify 通过，证明其实现的检查通过，不证明原交接所有要求已逐项满足。本次只读取并核对历史成果，没有重新训练、修改机制或为缺失的对齐测试补造结果。

## 9. 结论与适用范围

在冻结数据与各自预算下，本次结果支持“较强客户端更新高斯扰动伴随预测效用损失”的描述性结论；各指标的小噪声响应存在差异，不能简化为所有指标单调下降。该结论受具体模型、更新尺度、校准、数据预处理和项目组成约束，不用于证明模型间相同隐私预算下的优劣，也不用于宣称成员推断或梯度重建风险降低。

参数扰动没有保护样本数、客户端数、ALMITY 类别统计/聚合权重以及离线审计材料。配置的 `auxiliary_statistics_protected=false`、`formal_epsilon_delta_verified=false` 应继续保留。若后续需要形式化隐私结论，应明确保护对象、邻接关系、组合与会计机制；若需要严格基线等价结论，应补做同环境参数/概率对齐和随机流验证，形成新证据而非改写本批次记录。

## 10. 结果、源码版本与迁移材料

约定上游起点为 `7aa2df37f6cf3830a0f001e6e1a93d88d22297d9`。但服务器备份 Git HEAD 实际为 master，初始提交 `126ea193c9102a867aa9ec9dbaa60015be67373d`，不是一个已冻结的隐私扩展最终提交。运行级 `code_commit` 列填写的是上游起点，不能当作本次完整运行源码提交。当前以源码/配置逐文件 SHA-256 和完整迁移归档标识实际交付版本。

本次核对发现本地 ALMITY 验收器比服务器备份多了 NumPy<2 的 trapezoid/trapz 兼容适配；训练文件和协议与备份一致。这是验收工具差异，不应隐藏；若重现服务器验收，使用完整备份版本与其 NumPy 2.3.3 环境。

{archive_table()}

上表两份 tar.gz 和历史精简 ZIP 是独立归档，不内嵌于本次“含报告交接”ZIP。解压新 ZIP 后，将完整备份目录放在解压根目录下的 `服务器迁移备份_2026-11-24/`，即可沿表中相对路径访问；历史 ZIP 仅供版本追溯，需要时放在解压根目录。文档、数据包、脚本和结果的其他链接已包含在新 ZIP 内。

精简 ZIP 不含模型检查点和 prepared 数组，也不含 ALMITY 全部详细逐轮 JSON；其诊断仅为任务级汇总，不能替代原交接要求的逐轮诊断。全量模型重载验收必须使用完整迁移包，不能只解精简 ZIP 就执行并期待通过。

运行级、目标级与方法级结果目录为 [FLR 正式验收产物](experiments/privacy/flr-gaussian-extension-v1/runs/formal/)、[ALMITY 正式结果](experiments/privacy/almity-gaussian-extension-v1/runs/formal/)。FLR 原始任务在相应扩展目录的 runs/条件/目标/种子；ALMITY 在 runs/formal/条件/目标/种子，二者不可互换。

本报告生成所用的 [文档依据核对 JSON](实验报告与交接材料/文档依据核对.json) 保存本次任务矩阵、哈希、逐条件诊断和源码差异记录；[证据核对脚本](audit_report_evidence.py) 与 [文档生成脚本](build_experiment_documents.py) 可在当前工作目录重建文档。脚本不会替代模型验收器。

## 附录：主要结论与证据对应

| 结论 | 直接证据 | 解释边界 |
| --- | --- | --- |
| 两方法正式矩阵各 1800 个且到 100 轮 | 任务集合、逐任务 progress/metrics、本次核对 JSON | 不等于完整科研审计全部达标 |
| sigma_1p0 下项目平均 AUC 低于 none | study 运行级表、目标聚合、本报告表 5 | 描述性结果，无显著性/因果机制结论 |
| FLR 有概率饱和，ALMITY 部分校准反向 | 预测饱和字段、逐任务 calibration.slope | 不因异常删除种子或换评价口径 |
| 数据与划分冻结可追溯 | 数据包哈希、prepared 内容哈希、两方法划分 JSON | 不宣称跨源全局无特征重复 |
| 成果已可离线保管 | 完整项目/环境归档、本地与原服务器哈希一致记录 | Linux 环境恢复到新硬件仍需 smoke 和验收 |
"""

    handoff = f"""# FLR 与 ALMITY 隐私实验：开发、运行与迁移交接

版本：2026-11-24 / v1.0（实验完成后交接版）。本文件沿用原交接的 15 节结构和证据要求；范围收敛为本次负责的 FLR、ALMITY，并增加服务器关停后的恢复说明。

配套：[实验报告]({REPORT_NAME})；原标准：[隐私—效用权衡扩展实验开发与运行交接](隐私-效用权衡扩展实验_开发与运行交接.md)。原文件保持原样。本文中的“已完成”仅对所列证据范围负责，不以验收器 passed 代替所有要求。

## 1. 任务边界

本次任务是保留原学习器、预算和聚合机制，增加聚合前“客户端更新整体裁剪＋高斯扰动”，不是重跑同构方法排名，也不是原论文原生隐私机制复现。仅 FLR、ALMITY 各 1800 个正式模型，合计 3600 个；其他历史方法不计数。

当前正式运行已结束，已迁出完整项目、数据、结果、检查点及 FLR 环境。接手方应先保管备份、恢复环境并核验已有结果，不应因服务器停机就重新启动正式矩阵。若要补齐原交接尚未证实的要求，应在独立副本中进行，并保留原批次。

| 方法 | 正式模型 | 评价行 | 保存验收结果 |
| --- | --- | --- | --- |
| FLR | 1800/1800，全部 100 轮 | 3600（study/native） | passed，错误 0 |
| ALMITY | 1800/1800，全部 100 轮 | 5400（study/fixed_probability/native_margin） | passed，错误 0 |

## 2. 固定仓库版本与源码身份

上游仓库：[federated-defect-prediction-benchmark](https://github.com/yibingT-prog/federated-defect-prediction-benchmark)，约定共同起点 `7aa2df37f6cf3830a0f001e6e1a93d88d22297d9`。

重要版本差异：迁移归档内 `.git/HEAD` 指向 master，初始提交为 `126ea193c9102a867aa9ec9dbaa60015be67373d`。未见 privacy-ext/flr 或 privacy-ext/almity 已提交的最终扩展分支证据；不能只报上游起点，声称扩展源码已由该提交完整覆盖。以完整项目包与 [文档依据核对](实验报告与交接材料/文档依据核对.json) 中逐文件 SHA-256 作为交付身份。

若只 clone 上游冻结提交，只能取得原始基线/公共代码，拿不到本次未发布扩展的实际执行状态；优先使用已迁出的项目包。不要直接覆盖上游主分支、已发布同构结果或既有 FedCPDP 隐私结果。

## 3. 以哪些代码和文件为依据

### 3.1 FLR

基线：[experiments/homogeneous/flr-tff-20261111/](experiments/homogeneous/flr-tff-20261111/)。优先阅读其中 README、train_tff.py、protocol.json、tasks.json、splits、prepare_frozen.py、requirements-tff.txt 和 environment.freeze.txt。

扩展：[experiments/privacy/flr-gaussian-extension-v1/](experiments/privacy/flr-gaussian-extension-v1/)。关键入口为 train.py、privacy.py、run_study.py、verify.py、test_privacy.py、protocol.json。旧扩展 README 仍保留“待门禁验证”的历史描述，不能把它当作本次最终状态；以本交接、正式进度和验收记录为准。

### 3.2 ALMITY

基线：[experiments/homogeneous/almity-svm-20261114/](experiments/homogeneous/almity-svm-20261114/)。必须阅读 PUBLICATION.md、train.py 的 weights/gradient/execute、metrics.py、protocol.json、tasks.json、split_reference、prepare.py 和 check.py。它是声明过的线性 SVM 论文重实现，不是作者代码。

扩展：[experiments/privacy/almity-gaussian-extension-v1/](experiments/privacy/almity-gaussian-extension-v1/)。关键文件同上。服务器备份验收器与当前本地验收器只有已发现的 NumPy<2 trapezoid/trapz 兼容适配差异；训练代码、扰动模块、协议与备份逐字节一致。使用哪个版本必须记录，不将其混成同一个源文件哈希。

### 3.3 接手优先顺序

先读实验报告与本交接，再对照运行协议/任务/划分和原始输出，最后执行模型验收。不要把精简结果包中的任务级诊断 CSV，当成完整项目里每轮、每客户端的详细诊断。

## 4. 数据集与冻结划分

数据包：[Dataset.zip](Dataset.zip)，SHA-256 `4c19cbcf4a8d22fa99f47ec440b99dc716e6148a962d5af4f6f65292137b7801`。上游提交 `ab5af637a60aed4c34a5338709b3ac65b344ec37`，不得修改清单哈希来绕过错误。

{family_inventory()}

数据准备和恢复必须保留冻结源训练/验证行号、客户端名单、特征顺序及数据精度。源内部重复分组与源/目标同特征排除继续保留；跨源客户端训练/验证特征全局隔离没有约束。全目标测试记录保留，不删除 LOC=0 来修饰工作量指标。

本次核对：FLR 30 份 prepared 内容哈希全部匹配，ALMITY 30 份全部匹配；30 个目标的 splits 与 split_reference JSON 全部一致。FLR 为 float32，ALMITY 为 float64，不要为了跨方法“统一”改变已有基线。

目标标签仅用于最终测试，不校准、不选噪声或训练预算。目标无标签特征的本地 min-max 预处理权限沿用原基线。N/A、困难客户端和低分种子均保留。

## 5. 扰动扩展的统一定义与保护边界

固定六条件：none、clip_only、sigma_0p1、sigma_0p25、sigma_0p5、sigma_1p0。C=1.0；none 同时跳过裁剪和加噪；clip_only 只裁剪；四个非零条件逐坐标噪声标准差分别为 0.1、0.25、0.5、1.0。

```text
delta = theta_local - theta_global
flat = flatten_all_trainable_parameters(delta)  # 权重与偏置一起
clipped = flat * min(1, C / (L2_norm(flat) + 1e-12))
private_delta = unflatten(clipped + Normal(0, (sigma*C)^2 I))
```

在客户端本地训练后、进入加权聚合前扰动；不是逐层裁剪，不是特征/标签/损失扰动，也不是聚合后对服务器模型加噪。噪声是逐坐标标准差 sigma×C，不是方差，不另除客户端数，不做第二次未声明裁剪。

参数维数、更新尺度、训练轮数和聚合权重均影响实际保护分析；相同网格不代表相同隐私预算。继续声明辅助统计不受保护、未核验形式化 epsilon/delta。样本数、客户端数量、类别统计、聚合权重、离线种子、检查点和审计信息都不能视为安全公开部署机制。

## 6. 方法特定实现与不可静默变更项

### 6.1 FLR

使用 TFF 0.20.0 + TF 2.8.0，原生 LR 和样本数加权平均。非 none 条件自定义 WeightedAggregationFactory 把所有 trainable tensors 展平、裁剪、加噪，再交给原 MeanFactory；none 使用原默认聚合路径。模型没有 BatchNorm 等需另行处理的可训练外状态。服务器仍使用原 SGD 流程，不手动反转 TFF 原生 delta。

尚缺独立的服务器更新等价性/同环境原基线参数对齐报告；不能由当前源码形态或最终 AUC 相近替代这些证据。不要升级 TFF、改 sklearn/PyTorch LR 后仍称同一个原生执行版本。

### 6.2 ALMITY

保留线性 SVM、hinge+L2、SGD、自适应权重，以及源验证集 Sigmoid 校准与阈值选择。非 none 条件处理完整 `(lw−w, lb−b)` 后重建局部模型；none 不做减回加回式重建。聚合属性见逐任务 aggregation.json。

不因高噪声效用下降改 LR、学习率、聚合权重、过采样或校准器。校准概率评价与 raw margin 评价分别汇总；校准斜率为负不是应删掉的坏种子。

## 7. 训练预算与评价口径

| 设置 | FLR | ALMITY |
| --- | --- | --- |
| 实际聚合轮数 | 100 | 100 |
| 本地 epoch / batch | 100 / 20 | 5 / 32 |
| 学习率 | 客户端 0.1，服务器 1.0 | 0.01 |
| 优化器与初始化 | SGD、零初始化 | 无动量 SGD、零初始化 |
| 模型/目标函数 | 单层 Sigmoid LR、二元交叉熵 | 线性 SVM、hinge + L2/2×||w||²，L2=0.0001 |
| 参与率 | 1.0 | 1.0 |
| 模型种子 | 0–9，固定划分 | 0–9，固定划分 |

使用最后一轮。源项目宏平均 MCC 选 study 阈值，平局取 Recall 高、再取阈值低；网格 0.05–0.95、步长 0.025。主指标 ROC-AUC、MCC、Recall、G-mean，附 F1。连续分数计算 AUC，不用二值预测。

按每目标十种子均值/样本标准差，再项目等权汇总；方法表的标准差为目标均值之间的样本标准差。工作量指标有效项目 13/30，N/A 不填 0。5%、10%、20%、40% 预算分列；原交接尚未最终确定工作量指标在论文中的地位，本次作为补充保留。

相对 none 的变化为主；相对 clip_only 区分裁剪与加噪影响。不得按测试表现选最优噪声档、删目标/种子、提前终止或做跨方法相同隐私预算排名。

## 8. 服务器停机后的备份、环境与数据恢复

### 8.1 先确认材料齐备

{archive_table()}

上表是独立归档清单，两份大备份和历史精简 ZIP 均未嵌入本次“含报告交接”ZIP。完整备份目录应与解压后的本文同级，保留 `服务器迁移备份_2026-11-24/` 及原文件名；历史 ZIP 为版本追溯引用，需要时放在同级。其他文档、脚本、数据包和结果链接均可在新 ZIP 解压后直接访问。

原服务器与本地完整项目包、环境包的 SHA-256 曾逐一核对一致，tar 结构检查通过。精简 ZIP 适合阅读、分析和分发；完整迁移包才包含模型、prepared 数组、完整 ALMITY 逐轮诊断和历史材料。不要删除唯一完整备份来节省空间。

Windows 可以用 7-Zip 查看归档。恢复训练请在 Linux 中操作；Python 环境含 Linux 动态库/符号链接，不能当作 Windows 可执行环境。以下均为**接手后的 Linux 操作命令**，本次写文档没有在新机器上重新执行，不标成已测试恢复成功。

### 8.2 恢复项目和 FLR 环境

在新 Linux 主机确认目标路径为空后执行；若已有同名目录先另选位置，不能覆盖旧数据。BACKUP 指向迁移目录，不包含服务器口令。

```bash
BACKUP=/实际路径/服务器迁移备份_2026-11-24
sha256sum "$BACKUP/project-and-results.tar.gz" "$BACKUP/flr-tff020-environment.tar.gz"
test ! -e /root/federated-defect-prediction-benchmark || exit 1
test ! -e /root/miniconda3/envs/flr-tff020 || exit 1
test ! -e /root/Dataset.zip || exit 1
tar -xzf "$BACKUP/project-and-results.tar.gz" -C /root federated-defect-prediction-benchmark Dataset.zip
mkdir -p /root/miniconda3/envs
tar -xzf "$BACKUP/flr-tff020-environment.tar.gz" -C /root/miniconda3/envs
cd /root/federated-defect-prediction-benchmark
FLR_PY=/root/miniconda3/envs/flr-tff020/bin/python
"$FLR_PY" -c 'import sys,tensorflow as tf,tensorflow_federated as tff; print(sys.version,tf.__version__,tff.__version__)'
"$FLR_PY" experiments/homogeneous/flr-tff-20261111/smoke_tff.py
```

建议恢复原绝对前缀，以减少路径失效；新操作系统的 glibc、驱动等仍可能不兼容，所以必须先 smoke，不能因包完整就声称任意 Linux 可无修改恢复。实际备份 Python 3.8.20、TF 2.8.0、TFF 0.20.0、NumPy 1.21.6、pandas 1.3.4、SciPy 1.7.3、sklearn 1.0.2。

项目包还包含原服务器的 repo.zip、监控文件及面板数据，以上命令仅选择性恢复项目和 Dataset.zip；这些辅助历史文件仍保留在归档中。

### 8.3 恢复 ALMITY 环境

项目中 .venv-almity 的 Python 链接指向原 `/root/miniconda3/bin/python3.12`，完整迁移未包含整套基础 Miniconda。若新机器该路径不存在，不能宣称原 venv 开箱即用。安装 Python 3.12（原实际版本 3.12.3）后新建独立环境，按固定 requirements 安装，不覆盖归档内旧环境：

```bash
python3.12 -m venv .venv-almity-restored
ALMITY_PY="$PWD/.venv-almity-restored/bin/python"
"$ALMITY_PY" -m pip install -r experiments/homogeneous/almity-svm-20261114/requirements.txt
"$ALMITY_PY" -c 'import sys,numpy,pandas,scipy,sklearn; print(sys.version,numpy.__version__,pandas.__version__,scipy.__version__,sklearn.__version__)'
```

原实际 NumPy 2.3.3、pandas 2.3.3、SciPy 1.16.2、sklearn 1.7.2。重建可能需要联网下载依赖，完整归档不是一套承诺可离线安装所有系统依赖的镜像。若使用本地验收器的 NumPy<2 兼容补丁，要记录该源文件差异。

### 8.4 数据恢复或重建

完整项目包已有 prepared 数组，优先先核验它们；仅在独立副本缺数据或需要证明重建一致时运行准备脚本。下面的准备操作会写 prepared，ALMITY prepare 还会重建任务/审计 JSON，不能直接对唯一历史成果目录操作。

```bash
"$ALMITY_PY" experiments/homogeneous/flr-tff-20261111/prepare_frozen.py --archive /root/Dataset.zip
"$ALMITY_PY" scripts/prepare_almity_data.py --archive /root/Dataset.zip
"$ALMITY_PY" experiments/homogeneous/almity-svm-20261114/prepare.py
"$ALMITY_PY" experiments/homogeneous/almity-svm-20261114/check.py
```

上述 scripts/prepare_almity_data.py 在**完整项目迁移包**内；旧精简 ZIP 的打包清单没有包含仓库公共 scripts，不能把它当成只需解压精简包即可运行的命令。FLR 重建检查数组内容哈希；npz 容器字节哈希与数组内容哈希不是同一个概念，勿随意改值绕过原运行指纹。

### 8.5 环境记录缺口

本批次实际使用 Conda/venv，缺少容器镜像 ID，不能补写成 Docker 已验收。CPU/GPU 型号、操作系统发行版和精确实际 worker/环境变量快照待补；脚本里配置了线程限制不等于实际所有进程均被独立证实遵从。恢复新机器时保存系统、Python、依赖 freeze、硬件和线程信息，勿用新机器环境反填为原运行环境。

## 9. 目录布局与实际可用接口

以下布局以恢复后的仓库根目录为基准；本机相同相对路径位于 E:/study/software 下的 experiments。FLR 与 ALMITY 正式任务层级不同，必须保留：

```text
experiments/privacy/flr-gaussian-extension-v1/
  train.py  privacy.py  protocol.json  run_study.py  verify.py  test_privacy.py
  pilot/<condition>/<target>/<seed>/                 # 完整迁移包
  runs/<condition>/<target>/<seed>/                  # 正式任务
  runs/_scheduler/formal/progress.json
  runs/_scheduler/formal/logs/<condition>/<target>/<seed>.log
  runs/formal/verification.json  run_metrics.csv  target_seed_summary.csv
  runs/formal/method_summary.csv  family_summary.csv

experiments/privacy/almity-gaussian-extension-v1/
  train.py  privacy.py  protocol.json  run_study.py  verify.py  test_privacy.py
  runs/<condition>/<target>/<seed>/                  # 试运行
  runs/formal/<condition>/<target>/<seed>/           # 正式任务
  runs/formal/logs/<condition>/<target>/<seed>.log
  runs/formal/study_progress.json  verification.json
  runs/formal/run_metrics.csv  target_seed_summary.csv  method_summary.csv
```

原交接中示例 study.json、conditions/*/protocol.json、baseline_provenance.json 并未按该名字实现；本扩展以 protocol.json 内六条件和基线 tasks/splits 表达实际等价配置，不声称这些示例文件存在。

### 9.1 机制检查与正式结果重验

```bash
"$FLR_PY" experiments/privacy/flr-gaussian-extension-v1/test_privacy.py
"$ALMITY_PY" experiments/privacy/almity-gaussian-extension-v1/test_privacy.py
"$FLR_PY" experiments/privacy/flr-gaussian-extension-v1/verify.py --stage formal
"$ALMITY_PY" experiments/privacy/almity-gaussian-extension-v1/verify.py --stage formal
```

验收器会重写 runs/formal 中的汇总/报告，若要保留原验收文件，先在副本核验。二者都没有 --preflight 或 --stage pilot 的验收入口；ALMITY 的 run_study 也没有 --workers 和 --stage pilot，不能照抄原文的待开发 CLI 当成实际能力。

### 9.2 两目标完整轮次试运行（仅新副本需要时）

FLR 调度入口：

```bash
"$FLR_PY" experiments/privacy/flr-gaussian-extension-v1/run_study.py --stage pilot --targets ar5 CM1 --seeds 0 --workers 1 --rounds 100
```

ALMITY 通过单任务入口循环，省略 --smoke 与 --max-rounds，完整执行 100 轮：

```bash
for target in ar5 CM1; do
  for condition in none clip_only sigma_0p1 sigma_0p25 sigma_0p5 sigma_1p0; do
    "$ALMITY_PY" experiments/privacy/almity-gaussian-extension-v1/train.py --stage pilot --target "$target" --seed 0 --condition "$condition" || exit 1
  done
done
```

FLR --pilot 最后只保存完成标记和权重，直接 return，不生成试运行目标/验证预测，因此原要求的“完整试运行预测和阈值验收”仍须补齐独立检查；不能调用不存在的 verify --stage pilot。历史试运行已有产物，接手核验无需重复计入正式任务。

### 9.3 正式启动（历史矩阵已完成，勿盲目重跑）

若另开新版本并补齐门禁，现有接口为：

```bash
"$FLR_PY" experiments/privacy/flr-gaussian-extension-v1/run_study.py --stage formal --workers 2 --rounds 100
"$ALMITY_PY" experiments/privacy/almity-gaussian-extension-v1/run_study.py --stage formal
```

FLR 的 launch_formal_after_pilot.sh 绑定原服务器绝对路径且设置 workers=24，不是通用部署配置；不能不查新机器内存就照抄。推荐从 1 个任务验证内存再增至 2 个，而不是把历史监控脚本当作新机器默认资源设置。

## 10. 随机性、恢复和调度要求

### 10.1 实际随机流

ALMITY 训练种子为 SeedSequence([seed, round, client])；噪声另用 [seed, target_sha256前4字节, round, client, 991]。四个非零档不将 sigma 写入随机索引，因此同索引的标准正态样本配对后按 sigma 缩放。训练与噪声 RNG 分离，不同模型种子独立。

FLR 设置 tf.random.set_seed(seed)，数据 shuffle 按 seed×100000+round×100+client 派生；噪声为 TFF TensorFlow 计算中的有状态 tf.random.normal，没有实现独立的显式目标/轮次/客户端 stateless 随机流。不能宣称已证明不同客户端流无相关、训练/噪声完全独立、四档标准噪声严格配对或跨机器逐位确定。后续若修改随机机制，属于新代码/新实验身份，不能只替换部分高噪声任务。

### 10.2 中断与恢复

FLR latest_weights.npz 仅模型权重，不是完整 TFF/优化器/RNG 状态；当前训练入口从头初始化。未完成任务应隔离旧目录后从第 1 轮重跑，不能把最新权重称为等价续训检查点。

ALMITY 每轮原子保存模型、轮号与指纹，随机日程可由索引重建；但详细 diagnostics 只在整个训练循环结束后落盘。若中断在循环中间，checkpoint 可能已到较后轮而 diagnostics 不完整，续训可能出现不足 100 轮的诊断；尚无连续运行/中断恢复的独立等价测试。接手应先在副本测试，必要时把整个未完成任务隔离后从头重跑，保留旧检查点作审计。

两方法扩展均没有可靠的“放一个 STOP 文件即可安全暂停全部任务”的统一接口。ALMITY 基线中的 STOP 行为不能套用到扩展。停任务前确定调度器及其子进程的 PID/任务身份，停止后确认子进程退出；不要使用宽泛杀 Python 的命令影响无关工作。

### 10.3 调度与指纹

FLR 调度器优先凭输出身份/轮次跳过现有任务，并不在调度层核对完整代码指纹；训练入口有更深的指纹检查，但被调度器跳过时不会执行。ALMITY 训练入口检查已有结果/检查点指纹。两者未见逐任务互斥锁实现；FLR 先提交所有 futures，单任务失败后也不立即停止其余任务，ALMITY 顺序调度在失败时停止。

因此同一输出树只能有一个控制器。改变代码或数据前冻结旧结果，在独立新目录执行，先核验旧批次而不是依赖“metrics.json 存在”。当前指纹虽包含扩展训练/扰动/协议和部分基线代码、数据身份，未覆盖所有评价依赖/环境/调度代码；原要求的完整指纹闭包仍应补审。不得为了恢复绕过不匹配或删除唯一历史材料。

## 11. 开发与结果验收：已有证据和待补证据

| 原要求 | 当前证据/状态 | 接手动作 |
| --- | --- | --- |
| none 不裁剪不加噪 | 两训练入口显式旁路 | 保持原路径，不把 clip_only 当 none |
| 整体范数含偏置 | 两扰动模块可读；简单裁剪单测存在 | 补多张量、类型/形状、噪声期望样本及参数边界测试 |
| 原基线/none 逐轮或抽样参数、概率对齐 | 尚缺独立可复核报告 | 同环境同预算对齐，不仅看 AUC |
| TFF 更新符号与服务器等价性 | 使用原生服务器流程；未见独立等价测试 | 补小型可复核测试 |
| 独立噪声流、条件配对 | ALMITY 有显式索引；FLR 不能作同等保证 | 如需修复须冻结新版本和一致重跑 |
| 12 个完整试运行 | FLR 调度与完成标记 12/12；ALMITY 完整包有 12 个任务 | FLR 尚需预测、阈值、模型重载试运行证据 |
| 正式模型与预测验收 | 两正式 verification 均 passed、错误 0 | 完整包恢复后重载模型重验 |
| 任务无缺失/重复、轮数 | 本次独立核对 1800/方法、均 100 轮 | 保留核对 JSON |
| 准备内容与冻结划分 | 本次各 30/30 内容哈希，跨方法 30/30 索引一致 | 更换数据即另起版本 |
| 完整中断恢复、互斥锁、失败不扩散 | 当前尚有第 10 节缺口 | 不把现有调度描述为全要求通过 |

现有机制单测只是 [3,4]→[0.6,0.8]、none 和非有限输入等基础检查，不足以覆盖原要求的全部噪声、状态、不原地修改与非法配置测试。已有测试通过的事实可以报告，缺失的测试不能勾选通过。

正式 FLR 验收以最终权重重算目标/源验证概率、阈值、分类/工作量指标，预测容差为绝对 5.1×10⁻⁷，指标容差 10⁻¹⁰。ALMITY 目标/验证预测使用 rtol=10⁻¹²、atol=10⁻¹⁴，指标 10⁻¹⁰。均使用 pandas round_trip 解析。FLR 验收器注释中“pandas 默认六位小数”不能作为一般事实；实际 CSV 样例含更长精度，容差应视为该版本的核验设置，不据此虚构保存精度证明。

## 12. 数值问题、警告与常见误用

{diagnostic_table()}

FLR 总计 184 个任务至少一条预测饱和、78 个边界阈值；ALMITY 无饱和、64 个边界阈值。负校准斜率不是自动失败，校准概率和原生 margin 必须分开读。本次未建立完整警告类别/异常任务清单，不能把进程退出码 0 或验收 error_count=0 写成无警告、无校准问题。

逐任务目标预测保留原测试顺序 row_id。源验证 CSV 没有直接保存原 CSV 的 row_id，必须结合冻结 split 和数组顺序重建映射；FLR 目标 CSV 也没有单列 raw_LOC，重算工作量依赖 prepared/原始数据。现有验收主要比较数组顺序、标签和概率，不能额外宣称“验证 CSV 本身已带原行号和 LOC”。

ALMITY 完整包含每轮每客户端诊断；精简包只含任务级摘要。FLR 扰动工厂计算轮级诊断，但训练只每 5 轮把部分状态写入 progress 并最终用完成标记覆盖，未找到完整 100 轮独立诊断表，标准输出也未保留每轮完整 measurements。故 FLR 逐轮范数/裁剪率/噪声范数诊断仍为缺口，不能由最终饱和表替代。

不要：根据目标分数改噪声网格/学习率；静默删低分或 N/A；只在高噪声换稳定算子/校准器；用精简包运行依赖 checkpoint 的全量验收；将 5400 行误称 5400 个 ALMITY 模型；将代码表的上游 code_commit 当成最终运行源码提交。

## 13. 结果格式、文件清单与交付资料

### 13.1 任务身份与字段

唯一模型身份为 experiment_id + method + privacy_condition + target + seed + config_fingerprint。run_metrics.csv 包含 experiment_type、method、privacy_condition、threshold_mode、target、family、seed、rounds、clip_norm、noise_multiplier、code_commit、config_fingerprint 和各指标。threshold_mode 表示评价口径，不能与 privacy_condition 混用。

code_commit 的实际含义是声明的上游起点，不是隐私分支最终提交；完整源码身份见第 2 节和归档哈希。验证 CSV 原行号、FLR raw_LOC/逐轮诊断的限制见第 12 节，不将其写成已完整保存。

### 13.2 正式结果入口

- [FLR 运行级指标](experiments/privacy/flr-gaussian-extension-v1/runs/formal/run_metrics.csv)、[目标十种子均值/样本标准差](experiments/privacy/flr-gaussian-extension-v1/runs/formal/target_seed_summary.csv)、[项目宏平均与差值](experiments/privacy/flr-gaussian-extension-v1/runs/formal/method_summary.csv)、[家族表](experiments/privacy/flr-gaussian-extension-v1/runs/formal/family_summary.csv)、[验收 JSON](experiments/privacy/flr-gaussian-extension-v1/runs/formal/verification.json)。
- [ALMITY 运行级指标](experiments/privacy/almity-gaussian-extension-v1/runs/formal/run_metrics.csv)、[目标十种子统计](experiments/privacy/almity-gaussian-extension-v1/runs/formal/target_seed_summary.csv)、[项目宏平均与差值](experiments/privacy/almity-gaussian-extension-v1/runs/formal/method_summary.csv)、[验收 JSON](experiments/privacy/almity-gaussian-extension-v1/runs/formal/verification.json)。其原验收器未单独导出 family_summary.csv；本报告从 study 表按目标等权生成了家族 AUC，勿编造不存在的原文件。
- [FLR 正式调度进度](experiments/privacy/flr-gaussian-extension-v1/runs/_scheduler/formal/progress.json)、[ALMITY 正式进度](experiments/privacy/almity-gaussian-extension-v1/runs/formal/study_progress.json)。进度里的服务器绝对路径是历史字段；本地相应文件需按第 9 节布局定位，不把历史路径当本机路径。

### 13.3 回传资料与仍未证实项

| 原交付要求 | 本次材料 | 限制 |
| --- | --- | --- |
| 完整源码或最终分支提交 | 完整项目 tar.gz + SHA-256 | 未形成可核实的隐私最终分支提交 |
| 执行 README | 本交接第 8–10 节与实际入口 | 新机器恢复命令尚未实机复验 |
| 冻结配置、任务、数据、划分 | 基线目录 + 两扩展 protocol + Dataset.zip | 精简包不带 prepared 缓存 |
| 实际系统/依赖/硬件 | 原环境包、安装元数据、固定 requirements | 硬件/系统快照与容器 ID 缺口见第 8 节 |
| 结果、预测、日志、检查点 | 完整项目包 | 精简包主动不含 checkpoint；FLR 逐轮诊断缺失 |
| verification 与文件清单 | 两 verification、精简包 PACKAGE_MANIFEST.json、文档依据核对 | 现有清单不等于全量模型/环境每文件清单 |
| 机制改动、保护边界、异常 | 实验报告及本交接 | 不冒充正式 epsilon/delta 证明 |

完整备份共约 1.44 GB 压缩内容，超过此前精简包 100 MB 限制，二者作为不同交付层级分别保管。完整备份中含历史任务，不得和当前正式矩阵一起聚合。不要把服务器密码、密钥或令牌加入报告/交接或对外公开包。

## 14. 接手核对清单（保留原标准，不把缺口勾成通过）

- [x] 两方法正式任务各 1800、无缺失/重复、逐任务轮次为 100；现有验收 passed、错误 0。
- [x] Dataset.zip 哈希、30 目标准备数组内容哈希、跨方法冻结索引经过本次核对。
- [x] 六条件含义、含偏置整体裁剪和聚合前位置可由源码定位；主指标与原生评价分开保存。
- [x] 完整项目/检查点与 FLR 环境已从服务器迁出，并与原服务器归档 SHA-256 对照一致。
- [x] 饱和、边界阈值、ALMITY 负校准斜率和 LOC N/A 已在报告如实列明。
- [ ] 可核实的隐私分支最终提交与全部未提交修改清单；当前先用归档及逐文件哈希识别。
- [ ] 同环境原基线/none 参数与概率对齐，以及 TFF 服务器更新等价性独立报告。
- [ ] 原机制单测全部项目，而非现有少量基础测试。
- [ ] FLR 独立噪声流与多条件配对证明、完整逐轮范数/裁剪/噪声诊断。
- [ ] 完整的验证原行号发布、FLR 目标 raw_LOC 发布与聚合样本审计表。
- [ ] 经过中断等价测试的恢复流程、逐任务互斥锁和严格失败停止。
- [ ] FLR 两目标试运行的完整预测/阈值/模型核验材料，而非只看 12 个完成标记。
- [ ] 原机器完整系统/硬件/依赖实际快照，以及在新 Linux 机器实际执行恢复、smoke 和重验。
- [ ] 覆盖全部日志的异常/警告类别清单、全量源码/预测/检查点逐文件 SHA-256 清单。

这些待补项不改变已完成任务数量，但限制“严格满足原交接所有要求”的表述。补项应新增证据记录；涉及机制、随机性、预算、评价口径改变时，另起版本并一致验证，不把新旧批次拼为同一个正式矩阵。

## 15. 遇到问题时如何定位和回报

回报必须包含：方法、源码/归档 SHA-256、实际完整命令、操作系统和 Python/库版本、条件/目标/种子、最后完成轮数、错误日志路径、预期行为和是否使用精简包。不要只写“跑不了”。

| 现象 | 优先检查 | 处理原则 |
| --- | --- | --- |
| 找不到 checkpoint / prepared | 是否只解了约 78.5 MB 精简 ZIP | 用完整迁移包，或在副本严格重建输入 |
| 找不到 Python / bad interpreter | venv 链接依赖原基础 Miniconda 路径 | ALMITY 按固定依赖新建环境，不伪称可原地续训 |
| TF/TFF 导入或动态库失败 | Python 3.8、TF 2.8/TFF 0.20、Linux 系统库与恢复路径 | 先 smoke；不要静默升级 TFF |
| fingerprint 不匹配 | 训练/扰动/协议、基线代码、prepared 哈希与实际文件换行 | 保留旧包，定位差异，禁止改哈希绕过 |
| ALMITY diagnostics 少于 100 轮 | 中断时 checkpoint 与诊断落盘不同步 | 在副本测试恢复；必要时隔离旧任务后完整重跑 |
| 调度器立即跳过任务 | FLR 只按身份/轮次粗检跳过 | 先正式验收，不把跳过等于当前代码已重跑 |
| AUC 重算有差异 | round_trip 解析、保存精度、校准概率 vs raw margin | 按正确口径与原环境查因，不任意放宽容差 |
| 工作量指标 N/A | 原始 LOC 非正/缺失等前提 | 保留 N/A 和分类总体，不删行或填 0 |

若调整学习器、数据划分、C/sigma、训练预算、TFF 版本、校准/评价规则或噪声随机机制，先记录变更理由与新实验身份，再验证，不能以“修好”为由覆盖原正式产物。

交接目标是：接手方能在服务器已停的情况下找到正确材料、理解结果边界、恢复后核验模型，并明确哪些标准仍需补证。完整迁移成功不等于任何机器都已验证可续训；现有验收通过也不等于原 15 节全部要求已经满足。
"""
    (ROOT / REPORT_NAME).write_text(report, encoding="utf-8")
    (ROOT / HANDOFF_NAME).write_text(handoff, encoding="utf-8")
    # New documentation bundle preserves the delivered experimental package unchanged.
    source = ROOT / "FLR_ALMITY_隐私实验交付_最终版.zip"
    destination = ROOT / "FLR_ALMITY_正式交付_含报告交接.zip"
    additions = [ROOT / REPORT_NAME, ROOT / HANDOFF_NAME, EVIDENCE / "文档依据核对.json", ROOT / "audit_report_evidence.py", ROOT / "build_experiment_documents.py"]
    external_links = set(audit["archives"])
    links = []
    for name in (REPORT_NAME, HANDOFF_NAME):
        content = (ROOT / name).read_text(encoding="utf-8")
        for link in re.findall(r"\]\(([^)]+)\)", content):
            if link.startswith(("http://", "https://", "#")):
                continue
            assert (ROOT / link).exists(), (name, link)
            links.append(link)
    assert re.findall(r"^## (\d+)\.", handoff, re.M) == [str(i) for i in range(1, 16)]
    # Preserve original compressed entries; publish only after closing and checking
    # the appended archive. Testing an open append handle can report a false error.
    with tempfile.TemporaryDirectory(prefix=".document_bundle_", dir=ROOT) as temporary:
        draft = Path(temporary) / destination.name
        shutil.copyfile(source, draft)
        with zipfile.ZipFile(draft, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for path in additions:
                z.write(path, path.relative_to(ROOT).as_posix())
            supplemental = {"original_package_manifest": "PACKAGE_MANIFEST.json", "scope": "Original manifest covers original files only; this manifest covers added documentation", "external_large_backups_included": False,
                            "files": [{"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in additions]}
            z.writestr("DOCUMENT_MANIFEST.json", json.dumps(supplemental, ensure_ascii=False, indent=2))
        with zipfile.ZipFile(draft) as z:
            assert z.testzip() is None
            names = set(z.namelist())
            assert len(names) == len(z.infolist()), "Duplicate ZIP entries"
            for link in links:
                if link not in external_links:
                    assert link in names or any(n.startswith(link.rstrip("/") + "/") for n in names), link
            for item in supplemental["files"]:
                assert hashlib.sha256(z.read(item["path"])).hexdigest() == item["sha256"]
        assert draft.stat().st_size < 100_000_000
        draft.replace(destination)
    print(json.dumps({"report": REPORT_NAME, "handoff": HANDOFF_NAME, "handoff_numbered_sections": 15, "zip": destination.name, "zip_bytes": destination.stat().st_size, "crc": "passed", "relative_links": "passed; 3 documented external archives", "zip_sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
