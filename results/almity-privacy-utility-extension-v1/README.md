# ALMITY 隐私—效用扩展：正式发布结果

实验标识：`almity-gaussian-extension-v1`。共 30 目标 × 10 种子 × 6 条件 = 1800 个模型，均为 100 轮。

- [实验报告](../../FLR与ALMITY隐私实验_实验报告.md)、[运行与迁移交接](../../FLR与ALMITY隐私实验_开发运行与迁移交接.md)。
- [运行级结果](run_metrics.csv)、[目标统计](target_seed_summary.csv)、[方法统计](method_summary.csv)、[历史模型验收](verification.json)。
- [执行代码](../../experiments/privacy/almity-gaussian-extension-v1/)、[独立归档与数据](../../EXTERNAL_ARTIFACTS.md)。

FLR 保存 study/native，ALMITY 保存 study/fixed_probability/native_margin；评价行不等于模型数。完整预测、原始日志、prepared 和模型检查点保存在独立迁移归档，未加入 Git。本目录是训练结果的逐字节副本，保留历史 `code_commit` 与指纹，不回填本次发布提交来冒充原运行提交。`verification.json` 为历史模型验收证据；本次发布核对未重新训练或重载全部模型。

方法表先对每目标十种子求均值，再对目标等权平均；标准差为目标均值之间样本标准差。工作量指标有效目标仅 13/30。未检验形式化 epsilon/delta，原交接尚未补齐的证据见报告与交接。
