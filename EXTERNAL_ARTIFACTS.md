# 独立归档与数据说明

代码仓库提交源码、配置、冻结划分和整理后的指标；下列数据、压缩包、环境和模型通过此前本地交付独立保管，没有上传到 Git，也没有假定存在公开下载地址。报告中的链接先进入本清单，用于确认真实文件名与校验值。

<a id="datasetzip"></a>
## Dataset.zip

用户提供的原数据，SHA-256：`4c19cbcf4a8d22fa99f47ec440b99dc716e6148a962d5af4f6f65292137b7801`。数据来源提交：`MeeteshNevendra/FedCPDP@ab5af637a60aed4c34a5338709b3ac65b344ec37`。拿到同一数据包后先核验哈希，再按交接准备数据。原交接中的 Dataset.zip 指本地附件，GitHub 不内嵌原数据。

<a id="archive-1"></a>
## project-and-results.tar.gz

本地相对路径：`服务器迁移备份_2026-11-24/project-and-results.tar.gz`。大小：549361348 字节。SHA-256：`16e03f69e280d1e4be89ed13a8529beeadfcf97945dac0d2faa5824d16c1e75f`。

<a id="archive-2"></a>
## flr-tff020-environment.tar.gz

本地相对路径：`服务器迁移备份_2026-11-24/flr-tff020-environment.tar.gz`。大小：894972517 字节。SHA-256：`ce2970891c9943aa879c39d7c1926839d685f3f306775be85983d8b5e61772f6`。

<a id="archive-3"></a>
## FLR_ALMITY_隐私实验交付_最终版.zip

本地相对路径：`FLR_ALMITY_隐私实验交付_最终版.zip`。大小：78467621 字节。SHA-256：`d27318055182495a9ca27b10ce5498e8d501771acbc0dd918a81eaeda3f5692b`。

本地完整路径以交付目录为准；迁移备份位于 `服务器迁移备份_2026-11-24/`。项目包包含 prepared、完整预测/日志/检查点及 ALMITY 轮级诊断；精简 ZIP 不替代完整包。FLR 环境是 Linux Conda；ALMITY 原 venv 依赖未整套归档的基础 Python，恢复要求见交接。Git 仓库本身不足以重载历史模型完成全量验收。

两份历史文档生成/审计脚本也需要原独立归档和完整运行输出；仅 clone Git 仓库不能无输入重建这批历史报告。
