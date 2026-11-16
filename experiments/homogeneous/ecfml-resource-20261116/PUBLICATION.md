# ECFML 完整机制重实现：已完成发布

300 个正式任务、每次 50 轮，均于2026-11-16完成并核验。目标项目约80%有标签支持集适配，约20% query测试；不是目标完全无标签条件。 覆盖五个家族全部30个目标。

先阅读 [中文实验说明](实验说明.txt)、[实际协议](protocol.json)、[任务名单](tasks.json)、[结果](../../../results/ecfml-homogeneous-20261116/README.md)。README.md保留执行时说明；以下步骤适用于不包含原始数据和模型的Git发布版。

## 从仓库根目录运行

ECFML实际使用Windows、Python 3.13.5、PyTorch 2.13.0+cpu，其他依赖见environment.freeze.txt。 建议为该方法创建独立虚拟环境。以下安装CPU版本，不需要GPU：

```bash
python -m venv .venv-ecfml
# 按操作系统激活虚拟环境，再执行下面的命令。
python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r experiments/homogeneous/status.json/requirements.txt
python scripts/prepare_resource_data.py --method ecfml --download
python experiments/homogeneous/status.json/test_mechanisms.py
python experiments/homogeneous/status.json/verify.py --preflight
python experiments/homogeneous/status.json/schedule.py --workers 2
```

也可将`--download`替换为`--archive /path/to/Dataset.zip`。下载锁定FedCPDP作者提交`ab5af637a60aed4c34a5338709b3ac65b344ec37`，写入前核对全部30个CSV的SHA-256。数据不随仓库重新授权。

新的输出写入实验目录`runs`，调度器全部结束后自动执行`verify.py`；发布在`results`里的历史结果不会被覆盖。只有本地可信的检查点才能用于恢复。创建实验目录下的`STOP`后等待当前任务完成；删除STOP再启动调度器即可续跑。不要同时启动两个调度器。

## 执行快照与复现边界

训练代码、协议、任务、行号及ECFML本地重实现模块保持实际执行字节。`provenance`保留原包清单和来源哈希；发布版`package_manifest.json`仅删除未运行的FedDP作者参考文件条目，其他条目不变。该清单文件不参与训练指纹。作者参考源码不随发布版分发。

ECFML原指纹在Windows按本机路径排序，跨平台指纹和数值逐位一致不作保证；FedDP使用规范化相对路径排序。缺失数据时不能用正式训练指纹重放结果，先执行数据准备脚本。复现的学习器、预算及数据权限以protocol.json为准，而非原论文参考字段。

此代码来自本地论文机制重实现，不是作者公开仓库；保留MobileViT、FOMAML、裁剪/Laplace扰动和目标适配。未公开参数及目标批次BatchNorm统计访问已记录，不声称完整形式化隐私保证。
