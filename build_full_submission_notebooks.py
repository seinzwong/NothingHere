"""Publish executed bilingual notebooks only after the complete 50k study finishes."""
from pathlib import Path
import json,copy,uuid,textwrap,ast,shutil,subprocess,sys,hashlib,re,base64

ROOT=Path(__file__).resolve().parent
FULL=ROOT/'full_validation_study'
complete=json.loads((FULL/'complete.json').read_text(encoding='utf-8'))
assert complete['complete'] and complete['n_cka']==complete['n_insertion_deletion']==50000
source=(ROOT/'full_validation_pipeline.py').read_text(encoding='utf-8')
chunks={}
for piece in source.split('# %% ')[1:]:
    name,body=piece.split('\n',1)
    if name=='aggregation':body=body[:body.index('\ndef main():')]
    chunks[name]=body
chunks['environment']=chunks['environment'].replace("assert DEVICE.type=='cuda','Full study requires the verified CUDA environment.'", "if RUN_FULL_EXPERIMENT: assert DEVICE.type=='cuda','Full computation requires CUDA.'")

def cell(kind,text):
    text=textwrap.dedent(text).strip()+'\n'
    c={'cell_type':kind,'id':uuid.uuid4().hex[:8],'metadata':{},'source':text}
    if kind=='code':ast.parse(text);c.update(outputs=[],execution_count=None)
    return c

texts={
'zh':[
'''# ImageNet完整验证集：三种CNN的CKA与Grad-CAM扰动评估

**团队名称／成员／EID：请由团队提交前填写。**

本研究使用全部50,000张官方ImageNet验证图片（1,000类，每类50张），比较VGG16、ResNet50、MobileNetV2的七个原始卷积取样点。
CKA包含147个跨模型层组合；Insertion/Deletion同样使用全部50,000张图片，包含1,050,000组图片—模型—层组合和150,000组随机对照。
共有50,400,000个曲线采样点（共享端点不会重复计算）。没有将小样本结果冒充全量结果，没有训练或微调CNN。

问题一：不同架构的中间表示有多相似？问题二：各层Grad-CAM给出的空间排序能否比随机排序更快恢复／破坏固定类别分数？
两个问题互补：CKA不是解释图准确率，扰动AUC也不是分类准确率。

Notebook包含全部下载、环境设置、计算、断点续跑和分析代码，无需项目辅助脚本。默认读取完成后的缓存与图表；若要从头或续跑全量实验，设置`RUN_FULL_EXPERIMENT=True`。
首次完整运行需要已授权的ImageNet访问和数小时GPU时间；本机使用`Python (5070ti)`。只阅读嵌入输出无需下载图片。
''',
'''## 1. 环境与可复现性

使用三个模型各自的`IMAGENET1K_V1`预训练权重，固定种子5495，关闭TF32。架构是计算结构，权重是学到的参数；V1是权重版本，不是图片集文件名。
GAP特征和Grad-CAM采用FP32；CKA协方差采用FP64。扰动评分仅在实测精度检查通过后使用FP16，原始目标类别始终由FP32确定。
FP16通过范围是24张校准图片上的经验检查，不保证全验证集所有掩码的最坏误差；原图目标概率误差超过0.01的图片会改用FP32评分。
为提高长任务吞吐，I/D启用cuDNN算法选择，不承诺CUDA前向／反向逐bit复现；实际曲线和哈希均保存。

选择兼容的Python3.10环境。参考安装命令：`python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130`，并安装numpy、pandas、Pillow、pyarrow、huggingface_hub、matplotlib和ipykernel。
本notebook默认不会升级环境或自动安装包。`eval()`控制BN/Dropout行为，冻结参数防止训练，Grad-CAM仍需计算梯度。
''',
'''## 2. 数据与七层特征

只使用固定版本官方validation的全部14个parquet分片，约6.23GiB；不下载训练集。访问前在数据页接受条款并在本机执行`hf auth login`；token不写入代码或输出。
校验分片SHA256、图片解码、50,000条记录及每类50张。为保持“全量”范围，不因预测错误或源数据重复而排除图片；记录字节重复数。
图片转RGB、短边双线性缩放256、中心裁剪224，并使用ImageNet均值／标准差归一化。最终顺序由种子固定。

Tap是取样位置，hook在运行到该位置时记录输出。七点均为原始Conv2d输出，位于后续BN/ReLU之前；ResNet卷积点也在残差相加之前。
同槽位不表示语义或感受野相同。GAP把`[N,C,H,W]`的H、W平均成`[N,C]`，保留全部通道但丢失空间排列。
`clone()`防止inplace操作修改保存值。特征按小批量写入可恢复的磁盘数组，避免一次保留所有高分辨率特征图。
''',
'''## 3. 可扩展的精确Linear CKA

中心化后，`CKA=||Xc.T@Yc||F²/(||Xc.T@Xc||F·||Yc.T@Yc||F)`。
F范数是矩阵元素平方和开平方；每行是一张图，两个模型列数可以不同，但行必须对应同一图片。
本实现直接计算特征空间协方差，与样本Gram形式等价；不会建立50,000×50,000矩阵。它是普通、非去偏Linear CKA，不是minibatch CKA的简单平均。

比较全部7×7层组合，另对每类1、5、10、25、50张的均衡子集检查规模敏感性，并做10次单侧图片错配。
9个小规模真实层组合用另一种Gram公式复算。较高CKA表示GAP关系结构更相似，不代表准确率、因果依据或完整空间特征等价。
''',
'''## 4. 全量Grad-CAM与Insertion/Deletion

每张图／模型固定原始FP32完整输入的Top-1类别，对其logit求梯度；七层和全部扰动追踪同一目标，模型之间目标可能不同。
对通道梯度求空间均值，激活加权求和后整体ReLU，再双线性放大；随机优先级打破同分，全零CAM等于随机排序。
ReLU改为非inplace并检查前向等价，不更改权重。

Insertion从ImageNet均值颜色图开始逐步恢复像素；Deletion从原图开始逐步将像素替换为均值颜色。
归一化张量全零对应均值颜色，不是黑色。按排序使用20个区间、21个端点，记录固定目标softmax概率并做梯形积分。
**Insertion AUC高为好，Deletion AUC低为好**；不同于课堂概率差指标的方向。
Insertion改进=CAM−随机，Deletion改进=随机−CAM，正值均表示优于随机对照。
以类别为聚类单位做1,000次配对bootstrap，描述经验类别组成敏感性，不是完整有限验证集分数本身的测量误差，也未做多重比较校正。

完整原始卷积CAM以原空间分辨率分片保存，曲线、目标、全零标记和校验值同时保存。每64张形成一个检查点，意外中断可续跑。
FP16门限、FP32回退、CUDA非确定性和全量实际处理计数均写入审计记录。遮挡可能造成分布外输入，基线与步长都会影响结果；不能据AUC给模型准确率排名。
''',
'''## 5. 执行与完成状态

默认只读取通过完整性检查的全量结果。将开关设为True才会下载／续跑；这可能持续数小时。
只有数据、全部特征、CKA、三个模型的50,000张七层扰动评估与最终审计全部完成，才创建`complete.json`。
没有完成标记时本节明确停止，不把pilot或部分分片当成全量结果。
''',
'''## 6. 数值结果与图表

以下表格来自已完成的50,000张实验。CKA图中的浅／中／深只是相对采样组。
I/D图中的类别概率固定追踪原始Top-1，不能解释为各图地面真值类别的准确率。
配对bootstrap区间的单位是类别，未校正跨层多重比较。所有不利于Grad-CAM的结果也保留。
''',
'''## 7. 解释与限制

结论由下方实际数值生成。全部验证集减少了小样本覆盖问题，但不消除GAP信息损失、取样点角色差异、基线偏差、校准差异或扰动图分布外的问题。
CKA与I/D若给出不同排序，应分别报告，不能挑选有利指标。最高值是事后描述，不是预先指定的统计检验。
''',
'''## 8. 来源、提交与术语

缓存是保存以便复用的数值；manifest是来源清单；checkpoint是运行存档；seed固定随机过程；AUC是曲线下面积；logit是原始类别分数。
Linear probe另训练读出头，而本研究不训练CNN。Activation maximization合成人工图案；Grad-CAM解释空间响应；CKA比较表示几何。

本全量研究承接团队ImageNet模型配置。先前1,000张CKA／100张I/D作为pilot保留，不与本次全量数值混用。
主要提交是本notebook。两版代码与数值协议相同，已有输出可直接阅读；重新运行需要数据访问或原项目缓存。
参考：[CKA](https://proceedings.mlr.press/v97/kornblith19a.html)、[Grad-CAM](https://arxiv.org/abs/1610.02391)、[RISE扰动评估](https://arxiv.org/abs/1806.07421)、[ImageNet数据卡](https://huggingface.co/datasets/ILSVRC/imagenet-1k)。
'''],
'en':[
'''# Full ImageNet Validation: CNN Representation Similarity and Grad-CAM Perturbation Evaluation

**Team / Members / EIDs: to be completed before submission.**

All 50,000 official validation images (1,000 classes, 50 per class) are used for both Linear CKA and insertion/deletion.
VGG16, ResNet50, and MobileNetV2 each contribute seven raw-convolution taps: 147 cross-model CKA scores,
1,050,000 image-model-layer attribution evaluations, and 150,000 random-order controls.
The curves contain 50,400,000 sample points including shared endpoints, which are not redundantly evaluated.
No CNN is trained or fine-tuned, and pilot numbers are not substituted for the full benchmark.

Questions: How similar are representations across architectures, and do layer-specific Grad-CAM orders restore/suppress a fixed class score more effectively than random orders?
These are complementary questions; neither CKA nor perturbation AUC is classification accuracy.
All scientific code, including downloading and checkpoint recovery, is embedded below; no project helper script is required.
The default reads completed results. Set `RUN_FULL_EXPERIMENT=True` to compute or resume; initial execution needs authorized ImageNet access and hours of GPU time.
Saved outputs can be read without downloading images. The original machine uses the `Python (5070ti)` kernel.
''',
'''## 1. Environment and reproducibility

Use each model's explicit IMAGENET1K_V1 weights, seed 5495, and disabled TF32. Architecture specifies computation; weights are learned parameters.
GAP features and Grad-CAM use FP32; CKA covariance uses FP64. Perturbation scoring uses FP16 only after an empirical precision gate, while the target remains the FP32 original Top-1.
The 24-image calibration is not a worst-case guarantee over all masks; original-image target-score errors above 0.01 trigger FP32 scoring for that image.
cuDNN algorithm selection accelerates the long I/D run; bitwise CUDA forward/backward reproducibility is not claimed. Actual curves and hashes are retained.

Use a compatible Python 3.10 environment. Reference command: `python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130`, plus numpy, pandas, Pillow, pyarrow, huggingface_hub, matplotlib, and ipykernel.
This notebook does not automatically change the environment. Evaluation mode controls BatchNorm/Dropout; frozen weights prevent training, while Grad-CAM still requires gradients.
''',
'''## 2. Data and seven feature taps

Only the pinned official validation split is downloaded: 14 parquet shards, approximately 6.23 GiB, not the training set.
Accept ImageNet access conditions and authenticate locally with `hf auth login`; credentials are not embedded or printed.
Verify shard SHA256, decoding, exactly 50,000 records, and 50 images per class. Preserve every official row, including source duplicates, and report duplicate counts rather than silently excluding them.
RGB preprocessing uses bilinear shorter-edge resize to 256, a 224 center crop, and ImageNet mean/std normalization. Seeded final ordering is fixed.

A tap is a sampling location; a hook records its output. All taps are raw Conv2d outputs before subsequent BatchNorm/ReLU and, for ResNet, before residual addition.
Equal slots do not imply equal semantics or receptive fields. Full-channel GAP maps `[N,C,H,W]` to `[N,C]`, discarding spatial arrangement.
Immediate clones prevent inplace mutation. Features are stored in recoverable disk arrays rather than retaining all high-resolution feature maps.
''',
'''## 3. Scalable exact Linear CKA

After feature centering, `CKA=||Xc.T@Yc||F²/(||Xc.T@Xc||F*||Yc.T@Yc||F)`.
The Frobenius norm is the square root of summed squared entries. Rows correspond to the same images; column counts may differ.
Feature-space covariance is exactly equivalent to the Gram formulation without allocating any 50,000-by-50,000 matrix.
This is ordinary, non-debiased CKA, not a mean of minibatch CKA scores.

All 7-by-7 combinations are compared. Balanced subsets with 1, 5, 10, 25, and 50 images per class probe scale sensitivity; ten one-sided row mismatches provide controls.
Nine small real-feature combinations are independently recomputed using Gram arithmetic. High CKA concerns pooled sample relationships, not accuracy, causal evidence, or complete spatial equivalence.
''',
'''## 4. Full Grad-CAM and insertion/deletion

Fix each image/model's original FP32 Top-1 and differentiate its logit. Seven taps and all interventions retain that target; different models may have different targets.
Spatially average channel gradients, sum weighted activations, apply ReLU once, and bilinearly resize.
Seeded random priorities break ties; a zero CAM is identical to its random ordering. Non-inplace ReLU is checked for forward equivalence without changing weights.

Insertion restores ranked pixels from a mean-color image; deletion replaces ranked original pixels with mean color.
Zero normalized tensors correspond to ImageNet mean RGB, not black. Use 20 intervals, 21 endpoints, and trapezoidal integration of fixed-target softmax probabilities.
**Higher insertion AUC and lower deletion AUC are preferred**, unlike some probability-difference lecture metrics.
Insertion gain is CAM minus random; deletion gain is random minus CAM, so positive gains favor CAM in both cases.
One thousand paired class-cluster bootstrap resamples describe empirical class-composition sensitivity, not measurement uncertainty of the complete finite benchmark; intervals are not multiplicity-adjusted.

Store raw CAMs at their native spatial resolutions, with curves, targets, zero-map flags, and checksums. Commit a checkpoint every 64 images for recovery.
Precision gates, FP32 fallbacks, nondeterminism, and actual processed counts are audited.
Masking can produce out-of-distribution images; baseline, step size, and calibration matter. These AUCs are not a classifier-accuracy ranking.
''',
'''## 5. Execution and completion gate

The default reads verified full results. Enable the flag to download/compute/resume, potentially for hours.
`complete.json` is created only after all data, features, CKA, all three 50,000-image seven-layer I/D runs, and audits finish.
Missing completion metadata stops reporting rather than presenting pilot or partial shards as complete data.
''',
'''## 6. Numerical results and figures

The following outputs come from the completed 50,000-image experiment. Early/middle/late CKA groups describe sampling position.
I/D tracks the original Top-1 probability, not necessarily the ground-truth class.
Paired intervals resample classes and are not corrected for across-layer multiple comparisons. Unfavorable attribution results are retained.
''',
'''## 7. Interpretation and limitations

The text below is generated from measured values. Full validation removes the pilot's coverage limitation but does not remove GAP information loss,
operation-role differences, baseline effects, calibration differences, or out-of-distribution masking.
Report CKA and both perturbation directions separately when rankings disagree. Post-hoc maxima are descriptive, not prespecified tests.
''',
'''## 8. Sources, submission, and terminology

A cache stores reusable values; a manifest records sources; a checkpoint stores progress; a seed fixes random initialization; AUC is area under a curve; logits are unnormalized scores.
Linear probes train additional readouts, while this study does not train CNNs. Activation maximization synthesizes inputs, Grad-CAM highlights spatial responses, and CKA compares representation geometry.

The study follows the team's ImageNet model configuration. The earlier 1,000-image CKA / 100-image I/D remains a pilot and is not mixed with full-data scores.
This notebook is the primary submission. Both languages use identical scientific code/protocols and include saved outputs. Rerunning requires authorized access or project caches.
References: [CKA](https://proceedings.mlr.press/v97/kornblith19a.html), [Grad-CAM](https://arxiv.org/abs/1610.02391), [RISE](https://arxiv.org/abs/1806.07421), [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
''']}

def case_cells(lang):
    inputs=[]
    for key in ['dog','cat']:
        path=ROOT/'搭档已完成'/(key+'.png')
        inputs.append({'id':key,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                       'bytes_base64':base64.b64encode(path.read_bytes()).decode('ascii')})
    source=(ROOT/'cat_dog_case_study.py').read_text(encoding='utf-8').split('# CLI')[0]
    zh=lang=='zh'
    intro='''## A. 从队友的猫狗案例出发：解释图是否经得起扰动？

队友通过通道可视化、Grad-CAM和已训练的线性探针，说明具体图片在三种模型中的处理方式。本部分接着问：**优先恢复热力图认为重要的区域，能否更快恢复预测分数？优先删除这些区域，是否更快降低分数？**
使用队友提供的原始`dog.png`和`cat.png`，共两张案例图片，并非猫狗二分类数据集。模型仍输出ImageNet的1,000类；不重新训练CNN或线性探针。本节独立计算2×3×7=42组逐层评估和6组随机对照，后续50,000张主实验统计完全不变。

报告顺序：**猫狗个案 → 检查解释的扰动效果 → 扩展至完整ImageNet验证集 → CKA与总体扰动结果。**
''' if zh else '''## A. Starting from the team's dog and cat cases: testing the attribution ranking

The team's channel visualizations, Grad-CAM, and trained linear probes explain individual inputs. We extend those cases by asking: **does restoring highly ranked pixels recover the prediction faster, and does removing them reduce it faster?**
We use the two supplied original dog/cat PNGs. These are two illustrative inputs, not a two-class dataset; models still predict 1,000 ImageNet classes. No CNN or linear probe is trained. This separate case study contains 42 image-model-layer evaluations and six random controls; it does not alter the 50,000-image benchmark statistics.

Report sequence: **dog/cat cases → perturbation evaluation → complete ImageNet validation → CKA and aggregate perturbation findings.**
'''
    protocol='''### A.1 案例协议与可复现代码

与队友一致：保持长宽比、双线性缩放，居中补边到224×224，补边RGB=(124,116,104)，再按ImageNet均值／标准差归一化；使用相同V1权重、七个原始卷积取样点，并在每张图、每个模型内固定原始Top-1目标。目标可能因模型而异，不把“狗／猫”人为指定为目标，也不把预测当作人工真值。

Grad-CAM及扰动评分均用FP32，20个扰动区间、21个采样点，均值颜色背景，随机排序种子固定。案例使用CUDA；如记录反向算子非确定性警告，固定种子并不保证与队友CPU数组逐位一致。补边也参与像素排序与扰动，AUC针对整个224画布。
Insertion AUC越高越好；Deletion AUC越低越好；两种gain都定义为正值表示优于随机。热力图逐张归一化，颜色不能跨图比较强弱。**不为两个案例计算CKA**：本研究的样本级中心化线性CKA在两个非退化样本时会退化为1，缺乏区分意义。

原图字节保存在notebook元数据中，避免在报告中打印冗长编码；下方包含完整案例代码。默认读取已完成案例；`RUN_CASE_EXPERIMENT=True`才重新计算两张图，不重跑全量验证集。原始字节的SHA256与实际预处理几何保存在案例清单中。
''' if zh else '''### A.1 Case protocol and reproducible implementation

Match the team's aspect-preserving bilinear resize and centered 224-by-224 letterbox, RGB padding (124,116,104), ImageNet normalization, V1 weights, and seven raw-convolution taps. Each image/model fixes its own original Top-1 target, which may differ across models. Dog/cat labels are not forced targets or fine-grained ground truth.

Both Grad-CAM and perturbation scoring use FP32, 20 intervals/21 points, the mean-color baseline, and seeded random ranking. Cases run on CUDA; recorded nondeterministic backward operations mean fixed seeds do not promise bitwise agreement with the team's CPU arrays. Padding participates in ranking and perturbation, so AUC concerns the entire 224 canvas.
Higher insertion AUC and lower deletion AUC are better; both gain signs are positive for improvement over random. Each CAM is normalized separately. **No two-image CKA is computed:** sample-centered linear CKA degenerates to one with two nondegenerate samples under this representation setup.

Original image bytes are embedded in notebook metadata rather than printed as long encoded strings; the full case implementation follows. Read cached results by default; set `RUN_CASE_EXPERIMENT=True` to recompute only the two inputs. SHA256 and preprocessing geometry are recorded.
'''
    execute='''RUN_CASE_EXPERIMENT=False
# Locate the original PNG bytes embedded in this notebook's metadata.
CASE_INPUTS=None
for notebook_path in list(Path.cwd().glob('*.ipynb'))+list(ROOT.glob('*.ipynb')):
    embedded=json.loads(notebook_path.read_text(encoding='utf-8')).get('metadata',{}).get('case_input_data')
    if embedded:
        CASE_INPUTS=embedded
        break
if CASE_INPUTS is None:
    raise RuntimeError('Save this notebook with its embedded case_input_data metadata before running.')
'''+source+'''
case_folder=FULL/'cat_dog_cases'
if RUN_CASE_EXPERIMENT:
    run_cat_dog_cases(CASE_INPUTS)
case_record=json.loads((case_folder/'complete.json').read_text(encoding='utf-8'))
assert case_record['complete'] and case_record['n_images']==2
assert not case_record['included_in_full_validation_statistics']
for expected,saved in zip(CASE_INPUTS,case_record['inputs']):
    assert expected['sha256']==saved['sha256']
    assert digest(case_folder/(expected['id']+'_original.png'))==expected['sha256']
for record in case_record['models']:
    assert digest(case_folder/(record['model']+'.npz'))==record['sha256']
assert digest(case_folder/'summary.csv')==case_record['summary_sha256']
case_table=pd.read_csv(case_folder/'summary.csv')
assert len(case_table)==48
display(pd.DataFrame(case_record['inputs']))
display(pd.DataFrame([{'model':r['model'],'precision':r['precision'],'zero_maps':r['zero_maps'],
    'backward_warnings':r.get('backward_warnings',[])} for r in case_record['models']]))
display(DisplayImage(filename=str(case_folder/'case_inputs.png')))
for image_id in ['dog','cat']:
    display(Markdown('### '+image_id.title()))
    display(DisplayImage(filename=str(case_folder/(image_id+'_cams.png'))))
    display(DisplayImage(filename=str(case_folder/(image_id+'_curves.png'))))
    for model_name in SPECS:
        sub=case_table[(case_table.image==image_id)&(case_table.model==model_name)]
        display(Markdown('**'+model_name+' | '+str(sub.iloc[0].target)+'**'))
        display(sub[['method','original_probability','insertion_auc','deletion_auc',
            'insertion_gain_vs_random','deletion_gain_vs_random','all_zero']].round(4).reset_index(drop=True))
'''
    analysis="""
case_lines=[]
for (image_id,model_name),sub in case_table.groupby(['image','model'],sort=False):
    layers=sub[sub.method!='random']
    ins=layers.loc[layers.insertion_auc.idxmax()]
    dele=layers.loc[layers.deletion_auc.idxmin()]
    case_lines.append(f"**{image_id} / {model_name}**: target={sub.iloc[0].target}; best insertion {ins.method}={ins.insertion_auc:.4f}; best deletion {dele.method}={dele.deletion_auc:.4f}; layers better than random: insertion {int((layers.insertion_gain_vs_random>0).sum())}/7, deletion {int((layers.deletion_gain_vs_random>0).sum())}/7.")
display(Markdown('\\n\\n'.join(case_lines)))
"""
    if zh:
        analysis=analysis.replace('best insertion','Insertion最高').replace('best deletion','Deletion最低').replace('layers better than random: insertion','优于随机的层数：Insertion').replace('target=','目标=')
        analysis+="\ndisplay(Markdown('案例中的曲线可能起伏，逐步恢复并不保证概率单调上升，删除也不保证单调下降；应比较整条曲线的AUC。Insertion最佳层与Deletion最佳层可能不同，不能只选择有利指标。这里每个模型仅有两张图片和一份固定随机排序对照，不做总体显著性推断，也不依据跨模型的原始概率大小给模型排名。'))\n"
    else:
        analysis+="\ndisplay(Markdown('Perturbation probabilities need not change monotonically. Compare whole-curve AUC rather than selected endpoints. The best insertion and deletion layers can differ; report both metrics. With two inputs and one seeded random ranking per input, no population significance is claimed, and raw probabilities are not used to rank different models.'))\n"
    transition='''## B. 从两个案例扩展到完整ImageNet验证集

个案说明“这张图片上发生了什么”，不能说明总体表现。接下来固定相同模型、权重和七层取样位置，在官方验证集50,000张、1,000类上比较表示相似性与解释排序效果。

两部分的预处理有意区分：猫狗使用队友的letterbox保留整图；全验证集沿用预训练权重标准的resize/center-crop。案例使用FP32扰动评分，全量使用经过经验校准的FP16并保留FP32回退。协议差异意味着二者不是严格相同条件下的抽样对比，不能将差异全部归因于样本量。猫狗案例不加入全量均值或bootstrap，也不计算两图CKA。
下面完整保留既有全量实验的方法、代码、图表与结论。
''' if zh else '''## B. From two cases to complete ImageNet validation

Cases show what happens on particular inputs, not the population pattern. We now keep the same models, weights, and seven taps and evaluate representation similarity and attribution ranking across 50,000 validation images and 1,000 classes.

Cases use the team's full-image letterbox; the benchmark retains the weights' standard resize/center-crop. Case scoring is FP32, while benchmark scoring uses empirically gated FP16 with FP32 fallbacks. These protocol differences prevent a strictly controlled sample-size comparison; discrepancies cannot be attributed solely to sample count. Cases are excluded from full-data means/bootstrap, and no two-input CKA is computed.
The existing full-validation methods, implementation, figures, and conclusions follow unchanged.
'''
    return intro,protocol,[cell('code',execute),cell('code',analysis)],transition,inputs


def make(lang):
    t=texts[lang];cells=[cell('markdown',t[0]),cell('markdown',t[1]),cell('code','RUN_FULL_EXPERIMENT=False\n'+chunks['environment']+"\nfrom IPython.display import display,Markdown,Image as DisplayImage\nprint('Python:',sys.version,'torch:',torch.__version__,'torchvision:',torchvision.__version__)\nprint('Device:',DEVICE)"),
        cell('markdown',t[2]),cell('code',chunks['dataset']+'\n'+chunks['extraction']),
        cell('markdown',t[3]),cell('code',chunks['exact CKA']),
        cell('markdown',t[4]),cell('code',chunks['vectorized perturbations']+'\n'+chunks['full I/D']+'\n'+chunks['aggregation']),
        cell('markdown',t[5]),cell('code',"""
if RUN_FULL_EXPERIMENT:
    run_full_study()
completion_path=FULL/'complete.json'
if not completion_path.exists():
    raise RuntimeError('Full run incomplete: inspect full_validation_study/progress.json; no partial-data reporting.')
completion=json.loads(completion_path.read_text(encoding='utf-8'))
assert completion['complete'] and completion['n_cka']==completion['n_insertion_deletion']==50000
display(pd.DataFrame([{'CKA images':completion['n_cka'],'I/D images':completion['n_insertion_deletion'],
    'CKA scores':completion['cka_scores'],'Grad-CAM combinations':completion['gradcam_combinations'],
    'Random controls':completion['random_controls']}]))
print('Data manifest SHA256:',completion['data_manifest_sha256'])
display(pd.DataFrame(completion['precision']).T)
display(pd.DataFrame(completion['id_audit']))
"""),cell('markdown',t[6]),cell('code',"""
from IPython.display import display,Markdown,Image as DisplayImage
cka=pd.read_csv(FULL/'results/cka_main.csv')
id_table=pd.read_csv(FULL/'results/id_summary.csv')
assert len(cka)==147 and len(id_table)==24
display(cka)
display(id_table)
for figure in ['cka_full_matrices.png','cka_full_controls.png','id_full_curves.png','id_full_gains.png']:
    display(DisplayImage(filename=str(FULL/'figures'/figure)))
"""),cell('markdown',t[7])]
    if lang=='zh':
        analysis="""
lines=['# 全验证集结果解读']
for (a,b),table in cka.groupby(['model_a','model_b'],sort=False):
    matrix=table.pivot(index='slot_a',columns='slot_b',values='cka').to_numpy()
    i,j=np.unravel_index(np.argmax(matrix),matrix.shape)
    diagonal=np.diag(matrix)
    lines.append(f'**{a} / {b}**：最高为L{i+1:02d}—L{j+1:02d}，CKA={matrix[i,j]:.4f}；同槽位浅／中／深组均值为{diagonal[:2].mean():.4f}／{diagonal[2:5].mean():.4f}／{diagonal[5:].mean():.4f}。组均值不意味着逐层单调，也不表示语义对齐。')
for name in SPECS:
    sub=id_table[(id_table.model==name)&(id_table.method!='random')]
    random_row=id_table[(id_table.model==name)&(id_table.method=='random')].iloc[0]
    ins=sub.loc[sub.insertion_auc_mean.idxmax()];dele=sub.loc[sub.deletion_auc_mean.idxmin()]
    lines.append(f'**{name}**：Insertion最高{ins.method}={ins.insertion_auc_mean:.4f}，随机={random_row.insertion_auc_mean:.4f}；Deletion最低{dele.method}={dele.deletion_auc_mean:.4f}，随机={random_row.deletion_auc_mean:.4f}。分别有{int((sub.ins_gain_ci_low>0).sum())}／{int((sub.del_gain_ci_low>0).sum())}层的改进区间完全高于0。全零图共{int(sub.all_zero_maps.sum())}张，仍保留评估。')
lines.extend(['较高的浅层GAP相似度可能由全局颜色／亮度统计主导；本研究未做颜色消融，不能将其解释为所有边缘检测器相同。',
 'Insertion有效不保证Deletion有效。随机散点遮挡可能比连续区域遮挡更破坏纹理；这是待验证假设，不是已证实原因。',
 '验证集全量分数是固定有限数据上的测量；类别bootstrap区间表达类别组成敏感性，不能自动推广至所有未来图片或证明因果解释。',
 '24张图片的FP16校准是经验检查，其他遮挡的误差仍可能更大；精度记录与FP32回退数已在审计表给出。'])
display(Markdown('\\n\\n'.join(lines)))
"""
    else:
        analysis="""
lines=['# Full-validation interpretation']
for (a,b),table in cka.groupby(['model_a','model_b'],sort=False):
    matrix=table.pivot(index='slot_a',columns='slot_b',values='cka').to_numpy()
    i,j=np.unravel_index(np.argmax(matrix),matrix.shape)
    diagonal=np.diag(matrix)
    lines.append(f'**{a} / {b}**: maximum L{i+1:02d}—L{j+1:02d}, CKA={matrix[i,j]:.4f}; same-slot early/middle/late means are {diagonal[:2].mean():.4f}/{diagonal[2:5].mean():.4f}/{diagonal[5:].mean():.4f}. Group means do not establish monotonic depth trends or semantic alignment.')
for name in SPECS:
    sub=id_table[(id_table.model==name)&(id_table.method!='random')]
    random_row=id_table[(id_table.model==name)&(id_table.method=='random')].iloc[0]
    ins=sub.loc[sub.insertion_auc_mean.idxmax()];dele=sub.loc[sub.deletion_auc_mean.idxmin()]
    lines.append(f'**{name}**: highest insertion {ins.method}={ins.insertion_auc_mean:.4f}, random={random_row.insertion_auc_mean:.4f}; lowest deletion {dele.method}={dele.deletion_auc_mean:.4f}, random={random_row.deletion_auc_mean:.4f}. {int((sub.ins_gain_ci_low>0).sum())}/{int((sub.del_gain_ci_low>0).sum())} layers have entirely positive gain intervals. All {int(sub.all_zero_maps.sum())} zero maps remain evaluated.')
lines.extend(['Global color/brightness may dominate high early GAP similarity. No color ablation was performed, so identical edge detectors are not established.',
 'Insertion effectiveness does not guarantee deletion effectiveness. Random scattered masking may disrupt texture more than contiguous masking, but this proposed mechanism was not tested.',
 'Complete-validation metrics describe a fixed finite benchmark. Class-bootstrap intervals concern class-composition sensitivity and do not automatically generalize to all future images or establish causal explanations.',
 'FP16 calibration on 24 images is empirical; other masks can have larger errors. Precision measurements and FP32 fallback counts appear in the audit.'])
display(Markdown('\\n\\n'.join(lines)))
"""
    cells.extend([cell('code',analysis),cell('markdown',t[8])])
    intro,protocol,case_outputs,transition,case_inputs=case_cells(lang)
    cells=[cells[0],cell('markdown',intro),*cells[1:3],cell('markdown',protocol),
           cell('code',chunks['vectorized perturbations']),*case_outputs,cell('markdown',transition),*cells[3:]]
    cells[0]['source']=cells[0]['source'].replace('# ImageNet完整验证集：三种CNN的CKA与Grad-CAM扰动评估',
        '# 从猫狗案例到ImageNet完整验证集：三种CNN的CKA与Grad-CAM扰动评估').replace(
        '# Full ImageNet Validation: CNN Representation Similarity and Grad-CAM Perturbation Evaluation',
        '# From Dog/Cat Cases to Full ImageNet Validation: CKA and Grad-CAM Perturbation Evaluation')
    return {'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
        'kernelspec':{'display_name':'Python (5070ti)','language':'python','name':'5070ti'},
        'language_info':{'name':'python','version':'3.10.19'},'submission_language':lang,
        'scope':'Full ImageNet validation, 50,000 images for both CKA and insertion/deletion',
        'task_status':'Full computation complete; report execution pending','case_input_data':case_inputs}}

# Preserve completed pilot submissions before publishing the larger experiment.
pilot=ROOT/'submit/pilot_1000_100';pilot.mkdir(parents=True,exist_ok=True)
for path in [ROOT/'A2-cka-study.ipynb',ROOT/'submit/A2-cka-study-ZH.ipynb',ROOT/'submit/A2-cka-study-EN.ipynb']:
    if path.exists() and not (pilot/path.name).exists():shutil.copyfile(path,pilot/path.name)
zh,en=make('zh'),make('en')
for path,value in [(ROOT/'A2-cka-study.ipynb',zh),(ROOT/'submit/A2-cka-study-ZH.ipynb',copy.deepcopy(zh)),(ROOT/'submit/A2-cka-study-EN.ipynb',en)]:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=1),encoding='utf-8')
    subprocess.run([sys.executable,str(ROOT/'run_cka_notebook.py'),'--notebook',str(path)],check=True,cwd=str(ROOT))
validation={'complete':True,'n_cka':50000,'n_insertion_deletion':50000,'separate_case_images':2,'case_gradcam_combinations':42,'notebooks':[]}
validation['full_result_hashes_unchanged']=all(
    hashlib.sha256((FULL/'results'/name).read_bytes()).hexdigest()==expected
    for name,expected in complete['results_sha256'].items())
assert validation['full_result_hashes_unchanged']
images=[]
for path in [ROOT/'A2-cka-study.ipynb',ROOT/'submit/A2-cka-study-ZH.ipynb',ROOT/'submit/A2-cka-study-EN.ipynb']:
    raw=path.read_text(encoding='utf-8');n=json.loads(raw)
    assert not re.search(r'hf_[A-Za-z0-9]{20,}',raw)
    assert all(c.get('execution_count') is not None for c in n['cells'] if c['cell_type']=='code')
    assert not any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[]))
    n['metadata']['task_status']='Complete; all code cells executed, including separate dog/cat cases'
    path.write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
    for c in n['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))
    pngs=[hashlib.sha256(''.join(o['data']['image/png']).encode()).hexdigest() for c in n['cells'] for o in c.get('outputs',[]) if 'image/png' in o.get('data',{})]
    images.append(pngs)
    validation['notebooks'].append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'embedded_figures':len(pngs),'all_code_executed':True,'errors':0})
assert images[0]==images[1]==images[2]
validation['figures_identical_across_languages']=True
(ROOT/'submit/submission_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
(ROOT/'submit/README.md').write_text('''# 完整验证集提交版 / Full-validation submissions

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
''',encoding='utf-8')
resource_path=ROOT/'resource_status.json'
resource=json.loads(resource_path.read_text(encoding='utf-8')) if resource_path.exists() else {}
resource['full_validation_job']={'status':'complete','complete':True,'n_cka':50000,'n_insertion_deletion':50000,
    'results':'full_validation_study/complete.json','notebook_validation':'submit/submission_validation.json'}
resource_path.write_text(json.dumps(resource,ensure_ascii=False,indent=2),encoding='utf-8')
print('Published full-validation root, Chinese and English notebooks with executed results.',flush=True)
