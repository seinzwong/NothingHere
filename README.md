# XAI A2 — 从猫狗案例到完整 ImageNet 验证集

VGG16、ResNet50、MobileNetV2 的七层表示相似性（Linear CKA）与 Grad-CAM Insertion/Deletion 评估。模型使用 torchvision `IMAGENET1K_V1` 权重并保持冻结。

## 已完成的实验

| 模块 | 范围 | 输出 |
|---|---|---|
| 猫狗衔接案例 | 2 张原图 × 3 模型 × 7 层 | 42 组扰动评估、6 组随机对照 |
| 全量 CKA | 完整官方验证集 50,000 张、1,000 类 | 147 个跨模型层组合；1k/5k/10k/25k/50k 稳定性及打乱对照 |
| 全量 Insertion/Deletion | 50,000 张 × 3 模型 × 7 层 | 1,050,000 组层归因评估、150,000 组随机对照 |
| 曲线 | 每组 Insertion / Deletion 各 21 点 | 所有图片的完整曲线、逐图 AUC 和总体结果均已发布 |

**先阅读案例，再扩展至全量。** 两图案例单独统计；两样本 CKA 退化，因此不计算猫狗两图 CKA。案例采用队友的 letterbox 补边，全验证集采用标准 resize/center-crop；预处理及评分精度的差异已在报告说明，不能把结果差异全部解释为样本量效应。

## 报告入口

- [中文提交 notebook](submit/A2-cka-study-ZH.ipynb)
- [English submission notebook](submit/A2-cka-study-EN.ipynb)
- [主 notebook](A2-cka-study.ipynb)

三个文件已执行并嵌入 9 幅图，执行错误为 0；阅读输出无需下载数据。原图字节和案例科学代码已嵌入 notebook。提交前填写团队名、成员及 EID。当前仓库提供 notebook，尚未包含最终 PDF。

## 验证证据从哪里看？

| 证据 | 路径 | 能验证什么 |
|---|---|---|
| 全量完成清单 | [complete.json](full_validation_study/complete.json) | 实际范围、协议、模型与结果哈希、数值审计 |
| 原始数据清单 | [manifest.json](full_validation_study/data/manifest.json) | 50,000 行、类别、官方行号、原始图片 SHA256；每类 50 张 |
| 官方下载来源 | [download_manifest.json](full_validation_study/data/download_manifest.json) | 固定 HF commit、14 个官方分片大小与哈希 |
| 全部分片记录 | `full_validation_study/insertion_deletion/{model}/part_*.json` | 2,346 个断点：范围、原始 NPZ 校验值、全零图、FP32 回退、警告 |
| 完整曲线导出 | [curve_export.json](evidence/curve_export.json)、`evidence/curves/` | 无损导出全部 150,000 组图片—模型曲线，包含七层与随机对照 |
| 逐图 AUC | `full_validation_study/results/*_per_image_auc.npz` | 所有图片、层和随机对照的 AUC、标签与目标 |
| 数值复核 | [CKA audit](full_validation_study/results/cka_numeric_audit.csv)、[I/D audit](full_validation_study/results/id_numeric_audit.csv) | 小样本独立 Gram 复算、全部曲线梯形积分与端点核对 |
| 精度检查 | `full_validation_study/precision/` | FP16 对 FP32 的经验误差门限、实测误差 |
| 历史运行日志 | [run.log](full_validation_study/run.log)、[run.err.log](full_validation_study/run.err.log) | 原始分阶段时间记录、断电后续跑的记录及运行警告 |
| 提交文件检查 | [submission_validation.json](submit/submission_validation.json) | 执行状态、图表一致性、文件 SHA256、全量结果未改动 |
| 新的独立 CPU 复核 | [independent_verification.json](evidence/independent_verification.json) | 从公开曲线重算全部逐图 AUC、总体均值、随机对照差异及类别 bootstrap 区间 |

`progress.json` 仅为最终进度快照；历史阶段和断点在 `run.log` 及分片 JSON 中。日志不是完整终端录像，文件哈希也不是历史 GPU 执行的独立证明。仓库保存结果可复核性所需的完整曲线，但不会声称 CPU 复核重播了模型前向/反向。

## 不用 GPU 复核数值

Python 3.10+，安装 `numpy==2.1.2` 与 `pandas==2.3.3`（完整实测版本见 [environment_versions.json](evidence/environment_versions.json)），在仓库根目录执行：

```powershell
python verify_evidence.py
```

脚本验证数据清单和文件哈希、每类样本量、曲线索引连续性、概率范围与端点、全零 CAM 与随机曲线一致性，然后**重新计算全部 50,000×3×8×2 条曲线的 AUC、汇总分数和 1,000 次类别聚类配对 bootstrap 区间**，并检查提交 notebook 的执行记录。

CKA 的已保存数值和原始复核记录可以检查；由于原始 GAP 特征不分发，此 CPU 脚本不会重新计算全量 CKA。置信区间描述类别组成敏感性，不等于对完整有限验证集分数的测量误差，未做多重比较校正。

## 完整重新计算

原实验：Windows、RTX 5070 Ti 16GB、Python 3.10、PyTorch 2.9.1+cu130 / torchvision 0.24.1+cu130，内核 `Python (5070ti)`。环境精确版本见证据文件。需要已获授权的 Hugging Face `ILSVRC/imagenet-1k` 访问，且自行在本机登录；不要把 token 写入文件或提交仓库。

- notebook 默认 `RUN_FULL_EXPERIMENT=False`，读取已完成结果；设为 `True` 才重新计算/续跑全量实验。
- 猫狗案例用独立开关 `RUN_CASE_EXPERIMENT=True`，不触发全量重跑。
- CLI：`python -u full_validation_pipeline.py`。缺失的授权数据、官方模型及特征/扰动缓存会重建；原始全量计算需数小时和较大磁盘空间。首次环境请按 notebook 说明安装依赖。
- 一次只运行一个全量 CLI；代码包含 Windows 进程锁及断点恢复。数据版本、随机种子、取层、基线、步长与精度均写在配置中。

## 仓库边界

已包含全部代码、双语提交版、历史 pilot、全部全量指标与逐图曲线、猫狗原图与结果、2,346 个分片记录、运行日志及验证证据。

**未上传** ImageNet 原始 JPEG/Parquet、预训练模型权重、原始高维 CAM 与全量 GAP 缓存、HF 缓存/凭据、临时文件，以及队友报告和课程资料。ImageNet 通过官方授权来源获取。高维结果保留在本机；其大小与 SHA256 详见 [omitted_artifacts.json](evidence/omitted_artifacts.json)。这些缺省不会减少发布曲线的 50,000 张覆盖范围；仓库不宣称包含所有原始计算张量。

## Main interpretation

CKA measures similarity of pooled representations, not model accuracy or attribution quality. Insertion rewards rapid recovery (higher AUC); deletion rewards rapid score reduction (lower AUC). Both are compared against seeded random rankings. Cases explain individual behavior; the complete validation split tests wider coverage. No CNN is trained or fine-tuned. Full results, including unfavorable comparisons, are retained.

References: [CKA](https://proceedings.mlr.press/v97/kornblith19a.html), [Grad-CAM](https://arxiv.org/abs/1610.02391), [RISE](https://arxiv.org/abs/1806.07421), [ImageNet dataset card](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
