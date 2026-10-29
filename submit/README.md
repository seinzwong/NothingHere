# 完整验证集提交版 / Full-validation submissions

- `A2-cka-study-ZH.ipynb`：中文提交版。
- `A2-cka-study-EN.ipynb`：English submission.

两项实验均使用完整ImageNet验证集50,000张。包含147个CKA分数、1,050,000组七层Grad-CAM扰动评估及150,000组随机对照。
报告先展示队友同源猫狗两图的42组Grad-CAM扰动案例及6组随机对照，再扩展至全量结果；两图不计入50,000张统计。案例原图和重算代码已嵌入notebook。
两版已执行并保留输出；提交前填写团队、成员和EID。默认读取缓存；`RUN_FULL_EXPERIMENT=True`才会重新计算或续跑，耗时数小时。
完整科学代码嵌入notebook，无需项目辅助脚本。移动到新机器重算需要数据访问授权和适当环境。
`pilot_1000_100/`保留原小样本，不能与全量分数混用。

Both analyses use all 50,000 validation images. Outputs are executed and embedded; complete team details before submission.
The report starts with two separate dog/cat cases (42 layer evaluations and six random controls), followed by the unchanged full-validation study. Case inputs and computation code are embedded.
The default reads completed caches. Enable `RUN_FULL_EXPERIMENT` only to compute/resume the hours-long full study.
All scientific code is embedded; fresh computation needs authorized ImageNet access. Pilot results remain separately archived.
