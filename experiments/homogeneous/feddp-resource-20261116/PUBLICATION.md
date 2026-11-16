# FEDDP 完整机制重实现：已完成发布

230 个正式任务、每次 50 轮，均于2026-11-16完成并核验。独立公开项目用于无标签蒸馏，目标全部测试。公开项目从私人源客户端和目标名单中排除。 PROMISE 11、SOFTLAB 4、AEEEM 4、NASA 4个目标。ReLink只有3个项目，预留公开项目和目标后不足两个私人客户端，因此不适用；所有排除原因见 excluded.json。

先阅读 [中文实验说明](实验说明.txt)、[实际协议](protocol.json)、[任务名单](tasks.json)、[结果](../../../results/feddp-homogeneous-20261116/README.md)。README.md保留执行时说明；以下步骤适用于不包含原始数据和模型的Git发布版。

## 从仓库根目录运行

FedDP实际使用Ubuntu 24.04、Python 3.12.3、PyTorch 2.5.1+cpu，其他依赖见environment.freeze.txt与environment.system.txt。 建议为该方法创建独立虚拟环境。以下安装CPU版本，不需要GPU：

```bash
python -m venv .venv-feddp
# 按操作系统激活虚拟环境，再执行下面的命令。
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/homogeneous/status.json/requirements.txt
python scripts/prepare_resource_data.py --method feddp --download
python experiments/homogeneous/status.json/test_mechanisms.py
python experiments/homogeneous/status.json/verify.py --preflight
python experiments/homogeneous/status.json/schedule.py --workers 4
```

也可将`--download`替换为`--archive /path/to/Dataset.zip`。下载锁定FedCPDP作者提交`ab5af637a60aed4c34a5338709b3ac65b344ec37`，写入前核对全部30个CSV的SHA-256。数据不随仓库重新授权。

新的输出写入实验目录`runs`，调度器全部结束后自动执行`verify.py`；发布在`results`里的历史结果不会被覆盖。只有本地可信的检查点才能用于恢复。创建实验目录下的`STOP`后等待当前任务完成；删除STOP再启动调度器即可续跑。不要同时启动两个调度器。

## 执行快照与复现边界

训练代码、协议、任务、行号及ECFML本地重实现模块保持实际执行字节。`provenance`保留原包清单和来源哈希；发布版`package_manifest.json`仅删除未运行的FedDP作者参考文件条目，其他条目不变。该清单文件不参与训练指纹。作者参考源码不随发布版分发。

ECFML原指纹在Windows按本机路径排序，跨平台指纹和数值逐位一致不作保证；FedDP使用规范化相对路径排序。缺失数据时不能用正式训练指纹重放结果，先执行数据准备脚本。复现的学习器、预算及数据权限以protocol.json为准，而非原论文参考字段。

此代码依据作者可见实现重写为可运行CPU流程。修复路径、设备、密度张量和调度器缺参，并沿作者Client.train调用链使用过采样后的样本数聚合。MinMax预处理、单版本项目、源验证及10种子均为明确的协议适配。
