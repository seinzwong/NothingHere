"""Package Chinese and English notebooks with complete, executable methods."""
from pathlib import Path
import ast
import copy
import json
import re
import textwrap

ROOT=Path(__file__).resolve().parent
nb=json.loads((ROOT/'A2-cka-study.ipynb').read_text(encoding='utf-8'))
for c in nb['cells']:
    if isinstance(c['source'],list): c['source']=''.join(c['source'])
out=ROOT/'submit';out.mkdir(exist_ok=True)
cn=copy.deepcopy(nb)
cn['cells'][0]['source']=cn['cells'][0]['source'].replace('**我的贡献：**','**本模块：**')
cn['cells'][0]['source']+='\n**提交信息：团队名称、成员和EID请由团队在正式提交前填写。**\n'
cn['metadata']['submission_language']='zh-CN'

english_markdown={
0:r'''
# Comparing CNN Representations and Evaluating Grad-CAM on ImageNet
## Linear CKA, Insertion, and Deletion

**Team name / Members / EIDs: to be completed by the team before submission.**

This module compares seven intermediate convolutional representations in VGG16, ResNet50, and MobileNetV2 using Linear CKA,
then evaluates their Grad-CAM rankings using insertion/deletion perturbations. It complements the team's channel-synthesis,
two-image attribution, and linear-probe studies. No CNN or probe is trained here.

Research questions: (1) How similar are the sampled early, middle, and late representations across architectures?
(2) Do spatial rankings derived from those layers restore or suppress the original predicted-class score more effectively than random pixel orders?
Representation similarity and attribution evaluation address different questions; neither is classification accuracy.

CKA uses **1,000 ImageNet validation images, one per class**. Insertion/deletion uses the first 100 records in that fixed,
seed-shuffled subset, with no correctness or visual-quality filtering. Seven layers per model yield 147 cross-model CKA scores
and 2,100 image-model-layer perturbation evaluations, plus 300 image-model random controls.

Select **Python (5070ti)** or a compatible Python/CUDA environment. Run top to bottom. Resources are verified and reused when available.
When opened in `submit/`, this notebook finds the parent project's resources. In a fresh folder it downloads its own resources;
ImageNet requires local Hugging Face authentication and acceptance of the dataset's access conditions.
The notebook contains all experiment code and embedded results; no project helper script is required.
''',
1:r'''
## 1. Environment preparation and reproducibility

An architecture defines the computation; weights are learned parameters. `IMAGENET1K_V1` specifies a pretrained weight version,
not the name of a downloaded image file. GPU computation is used for the CNN, while CKA is computed on CPU in float64.
The tested environment is Python 3.10.19, torch 2.9.1+cu130, and torchvision 0.24.1+cu130 on an RTX 5070 Ti.

Dependencies are checked before execution. Package installation is disabled by default to preserve the existing environment;
enable it only when intentionally preparing a new compatible environment. The actual versions and hardware are recorded below.
The `Python (5070ti)` Jupyter kernel is registered on the original machine. Elsewhere, register the selected environment with
`python -m ipykernel install --user --name 5070ti --display-name "Python (5070ti)"`.

`eval()` controls evaluation behavior such as BatchNorm and Dropout. Freezing prevents parameter updates; `no_grad()` avoids
building a gradient graph during feature extraction. Grad-CAM later requires gradients, but still does not train the model.
''',
3:r'''
## 2. Models, sampling locations, and preprocessing

A *tap* is a chosen sampling location; a forward *hook* records a module's output. All seven taps below are **raw Conv2d outputs,
before subsequent BatchNorm/ReLU**. BatchNorm normalizes activations; ReLU removes negative values.
A ResNet conv3 output is also before BatchNorm and residual addition, so it is not the complete block output.
L01-L07 sample approximately evenly spaced main-path convolution indices; equal slots do not establish semantic equivalence.
The companion five-location visualization study mixes convolution, BatchNorm, and block outputs and is not substituted for this configuration.

All three V1 weights use the same deterministic preprocessing: RGB, bilinear resize of the shorter edge to 256,
224x224 center crop, rescale to [0,1], and ImageNet RGB normalization.
This differs from the letterbox preprocessing in the companion two-photograph study.
Global average pooling (**GAP**) averages spatial positions: `[B,C,H,W] -> [B,C]`.
Each row represents one image; channel counts may differ across models. GAP discards spatial arrangement,
so CKA conclusions concern pooled representations rather than every aspect of the complete feature maps.
''',
5:r'''
## 3. Authorized ImageNet access and subset construction

Only the official **ILSVRC/imagenet-1k validation split** is used, not training images or the unlabeled test split.
The dataset repository commit is pinned on first access and recorded in the manifest.
Accept the dataset access conditions in the browser and authenticate locally using `hf auth login`.
Credentials are read from local storage and are neither printed nor embedded. Missing authorization stops execution;
the notebook does not silently substitute CIFAR10 or fabricate results.

Streaming reads the fixed dataset version in its deterministic order. For each class, retain the first decodable image
whose exact bytes have not already been selected, until all 1,000 classes are covered; shuffle the retained records using seed 5495.
This is **sequential quota sampling, not uniform random sampling** of the validation set.
Extra records must be scanned, so network traffic exceeds the size of the saved subset. Byte hashing does not remove every near duplicate.
''',
7:r'''
## 4. Extracting and caching intermediate representations

An *inplace* operation changes existing storage. `detach().clone()` immediately copies a convolution's output so a subsequent
inplace activation cannot change the saved values. One forward per model and batch collects all seven taps.
GAP retains every channel: neither the 16 display channels nor probe-standardized features or class probabilities are substituted.

Caches are bound to image order, data and weight hashes, layer names, preprocessing, representation definition, and software versions.
Existence alone is insufficient for reuse. An initial batch checks the execution path before processing the complete subset.
Models are loaded sequentially to limit VRAM usage.
''',
9:r'''
## 5. Linear CKA and numerical checks

Center every feature column across images to obtain Xc and Yc:

\[
\operatorname{CKA}(X,Y)=\frac{\|X_c^T Y_c\|_F^2}{\|X_c^T X_c\|_F\|Y_c^T Y_c\|_F}.
\]

T denotes transpose; the Frobenius norm is the square root of the sum of squared matrix entries.
An equivalent implementation compares sample Gram matrices `K=Xc@Xc.T` and `L=Yc@Yc.T`.
Ordinary, non-debiased Linear CKA is between 0 and 1 for nondegenerate inputs; it is not accuracy and has no universal quality threshold.
One image becomes zero after centering; two images give CKA=1 whenever both representations are nondegenerate.
Per-channel standardization is different from centering and can change CKA.
''',
11:r'''
## 6. Cross-model results: three 7x7 matrices

Three model pairs and 49 layer combinations per pair yield **147 primary scores**.
All cross-slot combinations are included rather than presupposing layer alignment.
The matrix heatmaps visualize numerical representation similarity; they are not spatial Grad-CAM overlays.
Nine real early/middle/late combinations are additionally recomputed with the equivalent feature-space formula.
''',
13:r'''
## 7. Reliability: sample size and mismatched-image controls

Ordinary CKA can depend on sample count and feature dimension. Shuffling rows on only one side breaks image correspondence
and provides a diagnostic reference; its scores need not approach zero.
Ten mismatches are retained for every combination and are not presented as a formal significance test.

Nested subsets of 250, 500, and 1,000 images also change class coverage because the main sample has one image per class.
Consequently, changes are joint sample-size/coverage sensitivity, not an isolated sample-size effect.
A submatrix of the full Gram matrix must be recentered before calculating subset CKA; this is checked against direct recomputation.
''',
15:r'''
## 8. Interpretation generated from the measured CKA results

The following report uses measured values. Early/middle/late are **sampling groups**, not matched semantic stages or receptive fields.
The maximum of 49 combinations is a post-hoc descriptive result rather than a prespecified statistical test.
''',
18:r'''
## 9. Added question: do Grad-CAM rankings withstand perturbation evaluation?

**CKA compares representations; insertion/deletion evaluates attribution rankings.** The same ImageNet subset, preprocessing,
and seven raw-convolution taps are used. The first 100 images in the fixed order cover 100 different classes.
All three models and seven layers are evaluated: 2,100 image-model-layer combinations plus 300 image-model random controls.
Images are not filtered by correct prediction or selected for attractive visualizations. This is a budgeted exploratory sample,
not a representative evaluation of the complete ImageNet population.

### Grad-CAM, target, and ranking

For each image/model, fix the **original complete input's Top-1 class** and differentiate its logit.
Average gradients spatially to obtain channel weights, sum weighted raw activations, apply ReLU once, and bilinearly resize to 224x224.
The result is a spatial ranking, not a CKA heatmap. Targets remain fixed across all layers, perturbation steps, and controls;
different models may have different targets.

ReLU/ReLU6 are made non-inplace and forward equivalence is checked before differentiation, preventing storage mutation from corrupting taps.
The raw convolution locations remain before subsequent BatchNorm/ReLU.
CUDA backward runs with deterministic algorithms in **warn-only** mode: seeds and deterministic convolution settings are fixed,
but bitwise reproducibility of every backward operator is not claimed. Hash-verified cache recomputation is not an independent backward replay.

### Insertion, deletion, baseline, and AUC

- **Insertion:** start from a baseline image and restore pixels in descending attribution order; record the fixed target's softmax probability.
- **Deletion:** start from the original and replace the same ranked pixels with baseline values.
- **Baseline:** uniform ImageNet RGB mean `(0.485,0.456,0.406)`, equivalent to zeros after normalization; **not a black image**.
- **AUC:** trapezoidal integration of target probability against the actual fraction of pixels modified, including 0% and 100%:
  20 intervals and 21 sample points. **Higher insertion AUC and lower deletion AUC are preferred.**
  The lecture's probability-difference implementation is a different metric and its directions must not be transferred blindly.
- **Random control:** a seeded per-image pixel order shared across models. The same random priority breaks equal-CAM ties;
  an all-zero CAM therefore produces exactly the random order.
- **Paired gains:** insertion gain = CAM AUC minus random AUC; deletion gain = random AUC minus CAM AUC. Positive values favor CAM in both cases.
  Resample paired images 1,000 times and report 95% percentile intervals; no multiple-comparison adjustment or causal proof is claimed.

One shared mean-color baseline controls the protocol but does not reproduce every default in RISE.
Baseline choice, out-of-distribution masked inputs, step size, and spatial resolution can affect results.
An improved AUC only supports this perturbation protocol; it is not proof of a complete causal explanation or a cross-model accuracy ranking.
References: [Grad-CAM](https://arxiv.org/abs/1610.02391), [RISE / insertion-deletion](https://arxiv.org/abs/1806.07421).
''',
21:r'''
## 10. Insertion/deletion results, random controls, and uncertainty

The table reports mean AUC for each model's seven taps and its random control.
Paired bootstrap resamples images, keeping each image's CAM and random scores together.
Intervals describe resampling variation within these 100 exploratory observations, not all uncertainty from baseline choice,
gradient nondeterminism, or ImageNet representativeness.

Curves average the fixed-target probability across the 100 images. Negative gains, zero CAMs, and nonmonotonic curves are retained.
Insertion and deletion may prefer different layers; do not combine their AUCs as if they had the same direction.
''',
24:r'''
## 11. Perturbation interpretation and its relationship to CKA

The report below is generated from actual AUCs and paired intervals. Selecting a best layer among 21 is exploratory,
not a prespecified multiple-comparison-corrected finding. Different model Top-1 targets mean that cross-model AUC differences
are not matched-class explanation quality or classification-performance comparisons.

A high-CKA layer can have weak insertion/deletion results: CKA measures geometry across images, whereas these perturbations
test a class-specific spatial order. Early GAP may preserve similar global color statistics without ensuring precise Grad-CAM localization.
''',
26:r'''
## 12. Experiment record, terminology, and delivery

Recorded attempts: the default Python 3.7 lacked torch; the existing 5070ti environment passed CUDA checks.
CIFAR10 was considered and rejected to maintain the team's ImageNet route. Official weight and dataset transfers encountered network
interruptions, and ImageNet initially returned an authorization error until the user accepted the access conditions.
Only checksum-verified resources and successful executions are treated as completed work.

**Cache:** saved numerical results for reuse. **Manifest:** source/configuration record. **Seed:** a random process's fixed starting point.
**Checkpoint:** a training-state file, usually not an image dataset. **Logit:** an unnormalized class score.
**Linear probe:** an additional trained readout. **Grad-CAM:** class-specific spatial responses.
**Activation maximization:** optimize an artificial input to excite a unit. **CKA:** compare sample-relation structures.
**Insertion/deletion:** perturb a ranked input and track a fixed target score. **AUC:** area under that probability curve.

This notebook is the main submission, including executable code and embedded results. CSVs, figures, caches, and configuration files
are optional supporting artifacts for the team's reuse. Memes, slides, and video are separate team deliverables.
The Chinese and English notebooks use the same images, weights, layers, and numerical protocol.

References: [CKA paper](https://proceedings.mlr.press/v97/kornblith19a.html),
[author implementation](https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb),
[ImageNet access/data card](https://huggingface.co/datasets/ILSVRC/imagenet-1k),
[VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html),
[ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html),
[MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html),
[Grad-CAM](https://arxiv.org/abs/1610.02391), [RISE](https://arxiv.org/abs/1806.07421).
Local course background: Assignment2-2026A, Lecture5b, Lecture6a; companion model and two-image studies.
'''}

replacements={
'CNN参数保持冻结：CKA前向提取，Grad-CAM计算输入相关梯度；不训练CNN。':'CNN weights stay frozen: forward features for CKA and input-related gradients for Grad-CAM; no training.',
'当前解释器 {sys.executable} 缺少 {missing}；请选择5070ti环境，或在新环境启用安装开关。':'Interpreter {sys.executable} is missing {missing}; select the 5070ti environment or intentionally enable installation in a new environment.',
'项目输出根目录：':'Project output root:', '实际环境':'Observed environment',
'仅下载与前向；不训练CNN，不需要Lucent或Captum。':'Frozen models: forward feature extraction and input gradients for Grad-CAM; no CNN training or Lucent/Captum dependency.',
'官方API校验文件名给定的SHA256前缀；已缓存则不重复下载。':'Verify the official SHA256 prefix; reuse valid cached weights.',
'权重校验失败，请隔离损坏缓存后重新下载。':' weight verification failed; isolate the damaged cache and download again.',
'复用已校验的ImageNet子集：1000张，1000类。':'Reusing verified ImageNet subset: 1000 images, 1000 classes.',
'需要本机HF登录且接受ImageNet访问条款；不会切换为CIFAR10。':'Local HF authentication and ImageNet access are required; no CIFAR10 substitution.',
'固定版本未找到validation parquet，需核对仓库结构，不自动换数据。':'No validation parquet found at the pinned revision; inspect repository structure instead of substituting data.',
'只验证读取权限，不将带凭据的URL写入输出。':'Check read access without printing credential-bearing URLs.',
'流式记录没有图片字节，请核对数据格式。':'Streaming record contains no image bytes; inspect the data format.',
'已取得{len(chosen)}/1000类，扫描{scanned}张，耗时{time.monotonic()-started:.1f}秒':'Selected {len(chosen)}/1000 classes; scanned {scanned}; elapsed {time.monotonic()-started:.1f} seconds',
'仅取得{len(chosen)}类；不把不完整子集当成正式结果。':'Only {len(chosen)} classes obtained; the incomplete subset is not a completed experiment.',
'数据版本:':'Dataset revision:', '划分:':'split:', '每类1张，共':'One image per class; total',
'张；扫描记录数:':'images; records scanned:', '复用合格特征缓存:':'Reusing verified feature cache:',
'完成:':'Completed:',
'需要至少3行的有限二维矩阵；正式实验使用1000张。':'A finite 2D matrix with at least 3 rows is required; the main study uses 1000.',
'零变化表示或无效范数，CKA未定义。':'Constant representation or invalid norm: CKA is undefined.',
'样本行数不同。':'Different sample row counts.', 'CKA未定义。':'CKA is undefined.',
'9个真实浅／中／深组合经独立特征空间公式复算通过；最大误差：':'Nine real early/middle/late pairs passed feature-space recomputation; maximum error:',
'完整147个CKA数值（可折叠）':'All 147 CKA scores (collapsible)',
'子集表示退化。':'Degenerate subset representation.',
'验证子集Gram重中心化与直接重算一致。':'Check subset Gram recentering against direct recomputation.',
'错配对照完成':'Mismatch control completed',
'完成：21个层表示，147个主分数；公式检查通过；规模与错配对照完成。':'Completed: 21 layer representations, 147 primary scores, formula checks, size and mismatch controls.',
'附属结果目录：':'Supporting output directory:',
'重新运行优先校验缓存；所有结论均由真实结果生成。':'Reruns verify caches first; interpretations use measured results.'}

cka_english=r'''
lines=['# ImageNet Linear CKA: measured results and interpretation',
       f"Data: {REPO}, revision `{data_manifest['revision']}`, validation, one image per class (1000 total).",
       'Selection: first decodable, byte-unique image per class in the fixed stream order; not uniform random sampling.',
       'Models: three IMAGENET1K_V1 CNNs. Representation: full-channel GAP of seven raw convolution outputs. Estimator: ordinary centered Linear CKA.']
for (a,b),m in matrices.items():
    i,j=np.unravel_index(np.argmax(m),m.shape)
    diagonal=np.diag(m)
    groups=[float(diagonal[:2].mean()),float(diagonal[2:5].mean()),float(diagonal[5:].mean())]
    sh=np.stack(shuffle_matrices[(a,b)])
    gaps=diagonal-np.diag(sh.mean(axis=0))
    count=int(np.sum(diagonal>np.diag(sh.max(axis=0))))
    delta250=float(np.abs(stability[(a,b,250)]-m).max())
    delta500=float(np.abs(stability[(a,b,500)]-m).max())
    best_columns=m.argmax(axis=1)
    gains=m[np.arange(7),best_columns]-diagonal
    example=int(np.argmax(gains)); example_col=int(best_columns[example])
    increases=int(np.sum(np.diff(diagonal)>0))
    trend='higher' if groups[0]>groups[2] else 'lower' if groups[0]<groups[2] else 'equal'
    lines.extend([f'\n## {a} and {b}',
        f'Maximum: {SPECS[a][2][i]} (L{i+1:02d}) versus {SPECS[b][2][j]} (L{j+1:02d}), CKA={m[i,j]:.4f}.',
        f'Same-slot early (L01-L02), middle (L03-L05), and late (L06-L07) means are {groups[0]:.4f}, {groups[1]:.4f}, and {groups[2]:.4f}; the early mean is {trend} than the late mean.',
        f'L07 versus L07 is {m[6,6]:.4f}. The same-slot sequence increases at {increases} transitions, so the group trend is not a monotonic layer-by-layer decrease.',
        f'Cross-slot example: {a} L{example+1:02d} versus {b} L{example_col+1:02d} gives {m[example,example_col]:.4f}, compared with {diagonal[example]:.4f} at the matching slot. Slot-only comparisons miss some relationships; this does not establish semantic alignment.',
        f'Maximum absolute changes from the 1000-image matrix are {delta250:.4f} and {delta500:.4f} for 250 and 500 images. Class coverage changes too.',
        f'{count}/7 same-slot scores exceed the maximum of 10 mismatches. Gaps from mismatch means range from {gaps.min():.4f} to {gaps.max():.4f}; this is not a formal significance test.'])
all_first_max=all(np.unravel_index(np.argmax(m),m.shape)==(0,0) for m in matrices.values())
lines.extend(['\n## Reading these findings',
    'A CKA near 1 describes similar sample-relation structure, not percent classification accuracy or a percentage of identical neurons.',
    f'All three pairs attain their matrix maximum at L01-L01: {all_first_max}. Intermediate increases and cross-slot peaks prevent a universal monotonic-depth claim.',
    'Early raw convolution is approximately linear, and GAP removes spatial arrangement. Global color/brightness statistics may therefore drive high early similarity. This is an interpretation hypothesis, not a tested color-ablation result or proof of identical edge detectors.',
    f'Nine real early/middle/late pairs passed equivalent feature-space recomputation, maximum error {numeric_audit["absolute_error"].max():.2e}. This checks arithmetic on saved features, not an independent CNN replay.',
    '\n## Scope and limitations',
    'The quota sample has one image per class and cannot estimate within-class variation. Byte-duplicate exclusion is not near-duplicate removal.',
    'The compared slots have different operation roles, channel counts, and receptive fields; ResNet taps precede BatchNorm and residual addition.',
    'GAP discards spatial layout. Ordinary CKA can depend on dimension, sample size, and dominant variations. High mismatches or large subset changes call for more data and independent/debiased analyses.',
    'High CKA is not accuracy, fairness, or shared causal evidence; low CKA does not imply useless features.',
    '\n## Relationship to the perturbation study',
    'The CKA matrices quantify representation geometry. The following insertion/deletion study tests fixed-class spatial rankings. Their agreement or disagreement must be interpreted as complementary evidence, not an established causal relationship.'])
summary='\n\n'.join(lines)
display(Markdown(summary))
(OUT/'summary_en.md').write_text(summary,encoding='utf-8')
'''

id_english=r'''
id_lines=['# Grad-CAM insertion/deletion: measured results and interpretation',
    'First 100 records of the fixed ImageNet subset, with no correctness filtering. Each model/image fixes its original Top-1 target across all seven layers.',
    'ImageNet mean-color baseline; 21 sample points and probability AUC. Higher insertion and lower deletion AUC are preferred.']
all_layers=id_summary[id_summary['method']!='random']
id_lines.extend(['\n## Combined finding',
    f"Across 21 model-layer combinations, {int((all_layers.ins_gain_ci_low>0).sum())} insertion gain intervals are entirely above zero, compared with {int((all_layers.del_gain_ci_low>0).sum())} deletion intervals. Another {int((all_layers.del_gain_ci_high<0).sum())} deletion intervals are entirely negative. All are exploratory, unadjusted intervals.",
    'L07 has the highest insertion AUC within each model, but no layer has an entirely positive deletion gain interval. Some rankings support score recovery, but these results do not establish effectiveness under both perturbations.',
    'Scattered random masking may disrupt texture or push inputs off the natural-image distribution more than spatially contiguous masking. This is an explanatory hypothesis, not a verified mechanism; other baselines and patch-based interventions were not tested.'])
for name in SPECS:
    table=id_summary[(id_summary['model']==name)&(id_summary['method']!='random')]
    random_row=id_summary[(id_summary['model']==name)&(id_summary['method']=='random')].iloc[0]
    best_ins=table.loc[table['insertion_auc_mean'].idxmax()]
    best_del=table.loc[table['deletion_auc_mean'].idxmin()]
    wins_ins=int((table['ins_gain_ci_low']>0).sum())
    wins_del=int((table['del_gain_ci_low']>0).sum())
    id_lines.extend([f'\n## {name}',
        f'Random-order mean insertion/deletion AUC: {random_row.insertion_auc_mean:.4f}/{random_row.deletion_auc_mean:.4f}.',
        f'Highest insertion AUC: {best_ins.method} ({best_ins.insertion_auc_mean:.4f}); lowest deletion AUC: {best_del.method} ({best_del.deletion_auc_mean:.4f}).',
        f'{wins_ins}/{wins_del} layers have paired insertion/deletion gain intervals entirely above zero. These are exploratory, unadjusted 95% intervals.',
        f'The best sampled deletion layer has mean gain {best_del.deletion_gain_vs_random:.4f}, 95% interval [{best_del.del_gain_ci_low:.4f}, {best_del.del_gain_ci_high:.4f}]; only positive gain favors CAM over random.',
        f'Total all-zero maps across seven layers: {int(table.all_zero_maps.sum())}. They remain in the evaluation and are checked against their identical random-order curves.'])
id_lines.extend(['\n## Interpretation and limits',
    '1. Positive gains indicate faster score recovery/suppression relative to the chosen random order under this protocol, not improved accuracy.',
    '2. Report insertion and deletion separately when they select different layers; do not cherry-pick the favorable metric.',
    '3. Code checks fixed targets, original/baseline endpoints, pixel counts, AUC direction, non-inplace forward equivalence, and zero-map behavior.',
    '4. One mean-color baseline, 100 classes with one image each, 20 intervals, and one random order per image do not cover other baselines, step sizes, or all random orders. Bootstrap intervals omit those uncertainties.',
    '5. Masked images may be out of distribution. Calibration affects probabilities and model Top-1 targets may differ; these AUCs are not an accuracy ranking.',
    '6. Similar representation geometry need not imply identical attribution or perturbation performance. No causal link between CKA and AUC is established here.'])
id_text='\n\n'.join(id_lines)
display(Markdown(id_text))
(OUT/'insertion_deletion_summary_en.md').write_text(id_text,encoding='utf-8')
'''

en=copy.deepcopy(nb)
for index,source in english_markdown.items():
    assert en['cells'][index]['cell_type']=='markdown'
    en['cells'][index]['source']=textwrap.dedent(source).strip()+'\n'
for index,c in enumerate(en['cells']):
    if c['cell_type']!='code': continue
    if index==16:
        c['source']=textwrap.dedent(cka_english).strip()+'\n'
    elif index==25:
        tail=c['source'][c['source'].index('id_config='):]
        c['source']=textwrap.dedent(id_english).strip()+'\n'+tail
    else:
        for before,after in sorted(replacements.items(),key=lambda item:len(item[0]),reverse=True):
            c['source']=c['source'].replace(before,after)
    ast.parse(c['source'])
    if re.search(r'[\u4e00-\u9fff]',c['source']):
        remaining=[line for line in c['source'].splitlines() if re.search(r'[\u4e00-\u9fff]',line)]
        raise RuntimeError(f'Untranslated Chinese in English code cell {index}: {remaining}')
    # English outputs are regenerated by execution, not relabeled Chinese outputs.
    c['outputs']=[];c['execution_count']=None
en['metadata']['submission_language']='en'
en['metadata']['task_status']='English submission prepared; execution pending'
for name,n in [('A2-cka-study-ZH.ipynb',cn),('A2-cka-study-EN.ipynb',en)]:
    for c in n['cells']:
        if c['cell_type']=='code': ast.parse(c['source'])
    (out/name).write_text(json.dumps(n,ensure_ascii=False,indent=1),encoding='utf-8')
    print('Packaged:',out/name)
