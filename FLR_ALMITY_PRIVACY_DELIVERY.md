# FLR 与 ALMITY 隐私实验交付

分支：`privacy-ext/flr-almity-20261124`，共同起点：`7aa2df37f6cf3830a0f001e6e1a93d88d22297d9`。本次将已完成的历史实验整理为 Git 交付，不重新训练、不回填历史 code_commit、不修改原同构基线和 FedCPDP 结果。

- [实验报告](FLR与ALMITY隐私实验_实验报告.md)
- [15 节开发、运行与迁移交接](FLR与ALMITY隐私实验_开发运行与迁移交接.md)
- [FLR 正式结果](results/flr-privacy-utility-extension-v1/)
- [ALMITY 正式结果](results/almity-privacy-utility-extension-v1/)
- [独立归档与原数据](EXTERNAL_ARTIFACTS.md)
- [发布文件清单](FLR_ALMITY_PUBLICATION_MANIFEST.json)

每方法 1800 个模型，100 轮，历史模型验收 passed、错误 0。独立文档核对确认任务矩阵、冻结内容哈希和汇总统计。尚缺的原基线对齐、FLR 随机流/完整逐轮诊断等原要求如实保留为待补证，未因 Git 发布而宣称全部满足。

原执行源码在迁移包中的 Git 元数据不是完整隐私分支提交；本次提交是归档后的公开版本。除 README 状态说明和已披露的本地 ALMITY 验收兼容补丁外，扩展执行文件按迁移后的原字节发布。`FLR_ALMITY_PUBLICATION_MANIFEST.json` 记录源码和结果文件 SHA-256、基线关系及公开结果的来源映射；提交号从该分支 Git HEAD 读取，不写入自身制造自引用。

```bash
git clone --branch privacy-ext/flr-almity-20261124 https://github.com/three-penny/federated-defect-prediction-benchmark.git
cd federated-defect-prediction-benchmark
git rev-parse HEAD
```

原仓库写权限不足时，按原交接通过 `three-penny` 派生仓库分支交付。未向上游主分支推送，也未创建或合并 PR。
