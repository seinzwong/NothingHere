# 全量验证集任务：运行说明

目标已确定：**CKA与七层Insertion/Deletion都使用全部50,000张ImageNet验证图片**，不包含训练集。

这是持续数小时的后台计算任务。请以实时文件为准，不把现有pilot提交版当成全量结果：

- [实验阶段与计数](full_validation_study/progress.json)
- [监督进程状态](full_validation_study/supervisor_status.json)
- [运行日志](full_validation_study/run.log)
- [错误日志](full_validation_study/run.err.log)

执行顺序：14个官方分片下载及SHA256校验 → 50,000张图片检查 → 三模型全部GAP特征 → 精确特征空间CKA及对照 → 三模型全部七层Grad-CAM扰动评估 → 审计、图表 → 自动生成并执行中英文提交版。

已完成前置检查：GPU环境、三个V1权重、特征空间CKA与原147个分数的复算（最大差异约2.4e-15）、扰动方向／比例的解析玩具测试，以及FP16对照校准。
FP16只用于通过经验误差门限的扰动评分；Grad-CAM和原始目标由FP32产生，CKA使用FP64。
原图目标概率误差超过0.01的样本回退到FP32。校准不是所有遮挡输入的最坏误差保证。

下载支持断点续传；特征和I/D分片支持恢复。监督进程会监控任务，并在失败时有限重试；反复失败会明确标为`needs_attention`，不会发布虚假的完成结果。
任务期间监督进程防止系统自动睡眠，屏幕可以关闭；请保持主机开机与网络连接。无需持续打开聊天窗口。

只有`full_validation_study/complete.json`为真且提交版全部执行成功后，才算全量交付完成。
完成后会自动更新根目录的`A2-cka-study.ipynb`和`submit/`的中英文版；旧版归档到`submit/pilot_1000_100/`。

如需在进程确已停止后手动恢复，可在项目根目录运行：

```powershell
& 'C:\Anaconda3\envs\5070ti\python.exe' -u full_validation_pipeline.py
```

不要同时重复启动多个全量计算进程。当前是否运行请先看监督状态中的PID及实时日志。
