# FLR 隐私—效用扩展：交付入口

发布日期：2026-11-24。正式任务已完成（1800/1800、100 轮），现有模型验收 passed、错误 0。原要求中仍未补证的对齐、随机流、恢复等项目在交接中逐项列出，不能把正式结果验收等同全部门禁通过。

请从仓库根目录的 [开发运行与迁移交接](../../../FLR与ALMITY隐私实验_开发运行与迁移交接.md) 阅读安装、数据准备、机制检查、试运行、正式启动、停止/恢复和验收命令。结果入口为 [results/flr-privacy-utility-extension-v1](../../../results/flr-privacy-utility-extension-v1/)，报告见 [实验报告](../../../FLR与ALMITY隐私实验_实验报告.md)。训练仍输出到本目录下 runs，验收器也仍在 runs/formal 写汇总；发布时才复制到 results，未改变执行代码。

本次 Git 发布保持训练、扰动、协议和调度代码原始字节。ALMITY 本地验收器相对服务器版的 NumPy<2 兼容补丁见报告。完整数据和模型通过 [独立归档](../../../EXTERNAL_ARTIFACTS.md) 提供，不在 Git 中。

---

以下为原开发阶段 README，保留供追溯；其时间状态不代表当前结果状态。

# FLR Gaussian privacy-utility extension

This is an additive extension of the native TensorFlow Federated FLR baseline at
commit `7aa2df37f6cf3830a0f001e6e1a93d88d22297d9`. The six conditions apply one
global L2 clip over all trainable client deltas, including the bias, and add
Gaussian noise before weighted aggregation. `none` bypasses the perturbation
path. Formal epsilon/delta accounting is not claimed and auxiliary aggregation
statistics are not protected.

The extension remains gated on a TFF 0.20 compatibility test, baseline/`none`
alignment, and the two-target complete pilot before any formal matrix.
