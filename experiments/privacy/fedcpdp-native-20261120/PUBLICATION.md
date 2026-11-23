# FedCPDP 隐私—效用权衡实验

**实验类型：privacy_utility_tradeoff。不是同构方法对比实验，也不是通用超参数敏感度实验。**

本研究固定 FedCPDP 模型、训练预算、数据划分和评价规则，只改变原生更新裁剪与高斯扰动条件。底层数据场景仍是五个家族内的同构 LOPO；这描述数据，不决定研究问题。已有 `experiments/homogeneous/` 与对应结果不属于本批次。

2026-11-23 完成：6 条件 × 30 目标 × 10 种子 = **1,800 次新运行**，每次 50 轮。没有复用以前同构实验的模型或把它们拼成基线。

## 配置及范围

- `none`：不开启裁剪或噪声；`clip_only`：C=1，仅裁剪。
- `sigma_0p1`、`sigma_0p25`、`sigma_0p5`、`sigma_1p0`：C=1，σ 分别为 0.1、0.25、0.5、1.0。
- 五个家族内留一项目测试，源端约 80/20 冻结训练/验证划分，目标全部测试；种子 0–9。保持既定 v3 规则，不新增跨客户端重复分组。
- 50 通信轮、5 本地 epoch、batch 32，原生 MLP、哈希 1024、训练损失驱动的混合聚合。完整参数以各条件的 `protocol.json` 为准。
- 最后一轮模型；源验证客户端宏 MCC 选择阈值。目标标签不参与参数选择。目标无标签统计量参与原生标准化。
- σ 是噪声乘子，不是 ε。辅助损失与表示统计量未完整保护，没有端到端隐私会计或攻击风险实测。

六个试运行也是完整 50 轮，在同一运行包内计入正式矩阵。研究配置见 [study.json](study.json)，原执行说明见 [README.txt](README.txt)，[结果入口](../../../results/fedcpdp-privacy-utility-20261120/README.md)。

## 安装与运行（仓库根目录，Ubuntu）

CPU 足够，原运行环境为 Ubuntu 24.04、Python 3.12、4 vCPU、16 GiB；两个并行任务，每任务 BLAS 单线程。

```bash
python3.12 -m venv .venv-privacy
source .venv-privacy/bin/activate
python -m pip install -r experiments/privacy/fedcpdp-native-20261120/requirements.txt
python scripts/prepare_privacy_data.py --download
export PYTHONHASHSEED=0
python experiments/privacy/fedcpdp-native-20261120/validate_package.py
python -u experiments/privacy/fedcpdp-native-20261120/run_study.py --workers 2
```

已有作者数据压缩包时，用 `--archive /path/to/Dataset.zip` 替代 `--download`。准备程序先验证全部 30 个 CSV 的 SHA-256，再写入六个条件。数据不随仓库重新分发或重新授权。

仅检查六条件完整试运行：给 `run_study.py` 加 `--pilot-only`。它仍各训练 50 轮，不是短程 smoke test。新输出写到实验目录下，**不会覆盖 `results/` 中发布的历史产出**。

运行中查看 `study_status.json` 和各条件 `status.json`。停止时在六个 `conditions/<条件>/` 目录分别创建空文件 `STOP`，等待训练到轮次边界。续跑前移除 STOP，并使用相同命令；检查点按轮次恢复。不要同时启动两个调度器。异常中断产生的锁文件必须确认相关进程已退出后才可清理。

## 核验和报告重建

不需要原始数据或模型二进制即可复核公开预测、阈值、指标与汇总：

```bash
python scripts/verify_privacy_publication.py
```

重算统计和图表（约需数分钟）：

```bash
python -m pip install -r experiments/privacy/fedcpdp-native-20261120/analysis/requirements.txt
python experiments/privacy/fedcpdp-native-20261120/analysis/analyze_privacy_report.py
python experiments/privacy/fedcpdp-native-20261120/analysis/plot_privacy_report.py
python experiments/privacy/fedcpdp-native-20261120/analysis/build_privacy_report.py
```

报告脚本重建 Word 正文，分页与字体会受本机字体影响；已发布 Word 是经过原环境排版检查的版本。分析重建可能更新产物字节，验证发布哈希应针对未修改的 checkout。

## 执行证据与命名

原执行 `train.py` 及 `metrics.json` 中仍包含 `FedCPDP-author-native-homogeneous`，这是继承的学习器标识。为保持指纹和执行证据，**未改写历史源码、协议或结果字段**。这不是把本次实验归入同构对比：目录、发布元数据和合并结果都明确标记 `privacy_utility_tradeoff`。

原 CSV 的 `condition` 列表示 `study_mcc`、`fixed_05`、`validation_f1` 等阈值口径，**不是隐私强度**。跨条件合并必须另加 `privacy_condition`。发布的 `combined_all_runs.csv` 已区分这两个字段。

`package_manifest.json` 是执行时原清单，包含待重建数据条目；所有非数据条目保持原始字节。`protocol.environment` 中 Windows/Python 3.13.5 为继承模板字段，实际服务器运行 Python 3.12，依赖实录见 `provenance/environment.freeze.txt`。源端跨客户端重复、目标自标准化、原生近似梯度及高噪声数值饱和等限制见报告。
