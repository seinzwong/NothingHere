"""Extend the existing executed notebook; the deliverable remains self-contained."""
from pathlib import Path
import ast
import json
import textwrap
import uuid

ROOT=Path(__file__).resolve().parent
path=ROOT/'A2-cka-study.ipynb'
nb=json.loads(path.read_text(encoding='utf-8'))
for existing in nb['cells']:
    if isinstance(existing['source'],list): existing['source']=''.join(existing['source'])
if any(c.get('metadata',{}).get('section')=='insertion_deletion' for c in nb['cells']):
    raise RuntimeError('I/D section already exists; edit the existing section instead of appending twice.')

def cell(kind,source):
    source=textwrap.dedent(source).strip()+'\n'
    c={'cell_type':kind,'id':uuid.uuid4().hex[:8],
       'metadata':{'section':'insertion_deletion'},'source':source}
    if kind=='code':
        ast.parse(source);c.update(execution_count=None,outputs=[])
    return c

new=[]
new.append(cell('markdown',r'''
## 9. 新增研究问题：Grad-CAM 指出的区域经得起扰动检查吗？

**CKA比较模型表示，Insertion/Deletion评估解释图；两者回答不同问题。**
本节沿用前述ImageNet子集、预处理及七个原始卷积取样点。在固定顺序的前100张图（100个不同类别）上，
对三个模型各七层的Grad-CAM进行评估，共2,100组图片—模型—层组合，另加每模型每图一个随机排序对照。
不按预测是否正确筛选图片，不挑选视觉效果最好的例子。100张是计算预算下的探索性样本，不代表完整ImageNet。

### Grad-CAM、目标类别和排序

对每张图，固定该模型原始完整输入的Top-1类别`t`，对其 **logit** 求梯度；不同模型的`t`可能不同。
每通道的梯度空间均值作为权重，与原始激活加权求和后整体ReLU，再双线性放大至224×224。
这会把类别相关的空间响应转换为像素排序；它不是把CKA分数当成热力图。
七层、Insertion、Deletion和随机对照始终追踪同一个`t`，不会在每次遮挡后重新选Top-1。

先把所有ReLU/ReLU6设为非inplace，确认前向结果与原模型一致，再取梯度，避免原地修改破坏取样值。
卷积点仍在后续BN/ReLU之前。CUDA反向在`deterministic warn_only=True`下执行：固定种子与确定性卷积设置，
但不承诺全部梯度逐bit复现。缓存及其哈希用于复算，不能把复算缓存当成独立重放反向。

### Insertion／Deletion／AUC 是什么？

- **Insertion（插入）：**从基线图开始，按Grad-CAM由高到低逐步恢复原图像素；记录`t`的softmax概率。好的排序通常使分数较快恢复。
- **Deletion（删除）：**从原图开始，按同一排序逐步把像素替换成基线值；好的排序通常使分数较快降低。
- **基线（baseline）：**本节统一使用ImageNet均值RGB `(0.485,0.456,0.406)` 的纯色图，也就是归一化张量中的全零；**不是黑色图**。
- **AUC（曲线下面积）：**横轴是已插入／删除像素比例，纵轴是固定目标概率；包含0%和100%，20个区间、21个采样点，用梯形积分。
  **Insertion AUC越高越好，Deletion AUC越低越好。** Lecture6a使用的是另一种概率差定义，不能沿用其所有“高／低更好”方向。
- **随机对照：**每张图用固定种子的像素随机排序，同一排序复用于三个模型。像素同分时也用该随机优先级打破平局；全零CAM的排序因此等于随机对照。
- **配对改进：**Insertion改进=`CAM AUC−随机AUC`；Deletion改进=`随机AUC−CAM AUC`，两者正值均表示优于该随机排序。
  以图片为单位进行1,000次配对bootstrap，报告均值及95%百分位区间；它不是总体因果证明，也未做多重比较校正。

本节使用同一种均值基线，便于控制变量，并非复刻RISE默认的所有设置；基线改变、遮挡造成分布外图像、步长和解释分辨率都可能影响排名。
“AUC更好”仅说明在此扰动协议下更有效，不证明完整因果解释，也不用于跨不同目标类别给模型准确率排名。
参考：[Grad-CAM](https://arxiv.org/abs/1610.02391)、[RISE及Insertion/Deletion](https://arxiv.org/abs/1806.07421)。
'''))
new.append(cell('code',r'''
import warnings
import torch.nn.functional as F

ID_DIR=OUT/'insertion_deletion'
ID_DIR.mkdir(exist_ok=True)
ID_N=100
ID_STEPS=20
ID_BATCH=32
ID_REPEATS=1000
ID_RECORDS=data_manifest['images'][:ID_N]
ID_IMAGE_IDS=[r['image_id'] for r in ID_RECORDS]
assert len(ID_IMAGE_IDS)==len(set(ID_IMAGE_IDS))==ID_N
assert len({r['label'] for r in ID_RECORDS})==ID_N
PIXELS=224*224
ID_COUNTS=np.rint(np.linspace(0,PIXELS,ID_STEPS+1)).astype(np.int64)
ID_FRACTIONS=ID_COUNTS/PIXELS
assert ID_COUNTS[0]==0 and ID_COUNTS[-1]==PIXELS and np.all(np.diff(ID_COUNTS)>0)
ID_METHODS=[f'L{i:02d}' for i in range(1,8)]+['random']
ID_PROTOCOL={'n':ID_N,'selection':'first 100 records in the fixed seeded ImageNet subset order; no correctness filter',
    'image_ids':ID_IMAGE_IDS,'data_manifest_sha256':sha256_file(data_manifest_path),
    'target':'original full-input model Top-1; fixed across layers, perturbations and controls',
    'attribution':'Grad-CAM on original target logit; channel-mean gradient; weighted sum then ReLU',
    'map_resize':'bilinear 224x224 align_corners=False; nonnegative per-map maximum normalization',
    'baseline':'zero normalized tensor = ImageNet RGB mean, not black',
    'counts':ID_COUNTS.tolist(),'fractions':ID_FRACTIONS.tolist(),
    'score':'softmax probability of the fixed original target; not probability difference',
    'auc':'trapezoidal integral over actual pixel fractions; insertion higher, deletion lower',
    'tie_break':'per-image seeded random priority; all-zero map exactly equals random ordering',
    'seed':SEED,'random_order_seed_offset':20000,
    'cam_backward':'CUDA deterministic warn_only=True; bitwise reproducibility not guaranteed',
    'bootstrap_replicates':ID_REPEATS,'bootstrap_unit':'paired image',
    'torch':torch.__version__,'torchvision':torchvision.__version__,'preprocess':preprocess_config}

def random_priority(image_index):
    return np.random.default_rng(SEED+20000+image_index).permutation(PIXELS)

def ranked_pixels(cam,image_index):
    values=np.asarray(cam,dtype=np.float32).reshape(-1)
    assert values.size==PIXELS and np.isfinite(values).all()
    return np.lexsort((random_priority(image_index),-values))

def pixel_masks(order):
    assert np.array_equal(np.sort(order),np.arange(PIXELS))
    ranks=np.empty(PIXELS,dtype=np.int64)
    ranks[order]=np.arange(PIXELS)
    masks=ranks[None,:]<ID_COUNTS[:,None]
    assert np.array_equal(masks.sum(1),ID_COUNTS)
    return torch.from_numpy(masks.reshape(-1,1,224,224)).to(DEVICE)

def auc_probability(curve):
    return np.trapezoid(np.asarray(curve,dtype=np.float64),x=ID_FRACTIONS,axis=-1)

# Endpoint, ranking, and integration checks verify the perturbation protocol.
test_order=ranked_pixels(np.zeros((224,224),np.float32),0)
assert np.array_equal(test_order,np.argsort(random_priority(0)))
test_mask=pixel_masks(test_order)
assert not test_mask[0].any().item() and test_mask[-1].all().item()
assert np.isclose(auc_probability(np.ones(21)),1)
assert np.isclose(auc_probability(ID_FRACTIONS),0.5)
assert np.isclose(auc_probability(1-ID_FRACTIONS),0.5)
del test_mask
print('Protocol checks passed: masks, endpoints, constant/linear AUC, zero-CAM random equivalence.')
display(pd.DataFrame([{'quantity':'I/D images','value':ID_N},
    {'quantity':'Grad-CAM image-model-layer combinations','value':ID_N*3*7},
    {'quantity':'Random image-model controls','value':ID_N*3},
    {'quantity':'Curve samples including endpoints','value':ID_STEPS+1}]))
'''))
new.append(cell('code',r'''
id_data={}
for name,(builder,weights,layers) in SPECS.items():
    signature={'protocol':ID_PROTOCOL,'model':model_resources[name]}
    cache=ID_DIR/f'{name}_id.npz'
    meta=cache.with_suffix('.json')
    if cache.exists() and meta.exists():
        saved=json.loads(meta.read_text(encoding='utf-8'))
        if saved['signature']==signature and saved['sha256']==sha256_file(cache):
            with np.load(cache,allow_pickle=False) as f: arrays={k:f[k].copy() for k in f.files}
            assert arrays['image_ids'].tolist()==ID_IMAGE_IDS
            assert arrays['curves'].shape==(ID_N,8,2,ID_STEPS+1)
            id_data[name]=arrays
            print('Verified I/D cache:',name)
            continue
    started=time.monotonic()
    model=builder(weights=weights,progress=False).eval().to(DEVICE)
    for p in model.parameters(): p.requires_grad_(False)
    first=dataset[0][0].unsqueeze(0).to(DEVICE)
    with torch.no_grad(): original_output=model(first)
    for module in model.modules():
        if isinstance(module,(torch.nn.ReLU,torch.nn.ReLU6)): module.inplace=False
    with torch.no_grad(): adjusted_output=model(first)
    equivalence_error=float((original_output-adjusted_output).abs().max().item())
    assert torch.allclose(original_output,adjusted_output,atol=1e-6,rtol=1e-6)
    del original_output,adjusted_output,first
    cams=np.empty((ID_N,7,224,224),dtype=np.float32)
    targets=np.empty(ID_N,dtype=np.int64)
    original_prob=np.empty(ID_N,dtype=np.float64)
    warnings_seen=set()
    modules=dict(model.named_modules())
    grad_started=time.monotonic()
    old_deterministic=torch.are_deterministic_algorithms_enabled()
    old_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True,warn_only=True)
    try:
        for image_index in range(ID_N):
            x=dataset[image_index][0].unsqueeze(0).to(DEVICE).requires_grad_(True)
            activations={}; snapshots={}; handles=[]
            def capture(slot):
                def hook(module,inputs,output):
                    activations[slot]=output
                    snapshots[slot]=output.detach().clone()
                return hook
            for slot,layer in enumerate(layers):
                handles.append(modules[layer].register_forward_hook(capture(slot)))
            try:
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    logits=model(x)
                    target=int(logits.argmax(1).item())
                    gradients=torch.autograd.grad(logits[0,target],[activations[i] for i in range(7)])
                for w in captured: warnings_seen.add(str(w.message).split(' (')[0][:300])
                targets[image_index]=target
                original_prob[image_index]=logits.softmax(1)[0,target].item()
                for slot,(activation,gradient) in enumerate(zip([snapshots[i] for i in range(7)],gradients)):
                    raw=F.relu((gradient.mean((2,3),keepdim=True)*activation).sum(1,keepdim=True))
                    scaled=F.interpolate(raw,size=(224,224),mode='bilinear',align_corners=False)[0,0]
                    maximum=scaled.max()
                    if maximum.item()>0: scaled=scaled/maximum
                    cams[image_index,slot]=scaled.detach().cpu().numpy()
                assert np.isfinite(cams[image_index]).all()
            finally:
                for handle in handles: handle.remove()
            del x,logits,gradients,activations,snapshots
            if (image_index+1)%25==0: print(name,'Grad-CAM',image_index+1,'/',ID_N,flush=True)
    finally:
        torch.use_deterministic_algorithms(old_deterministic,warn_only=old_warn_only)
    grad_seconds=time.monotonic()-grad_started
    curves=np.empty((ID_N,8,2,ID_STEPS+1),dtype=np.float64)
    baseline_prob=np.empty(ID_N,dtype=np.float64)
    with torch.no_grad():
        for image_index in range(ID_N):
            x=dataset[image_index][0].unsqueeze(0).to(DEVICE)
            target=int(targets[image_index])
            baseline_prob[image_index]=model(torch.zeros_like(x)).softmax(1)[0,target].item()
            full_prob=model(x).softmax(1)[0,target].item()
            assert abs(full_prob-original_prob[image_index])<2e-5
            for method_index in range(8):
                order=(ranked_pixels(cams[image_index,method_index],image_index)
                       if method_index<7 else np.argsort(random_priority(image_index)))
                masks=pixel_masks(order)
                perturbed=torch.cat([x*masks,x*(~masks)],dim=0)
                scores=[]
                for batch in perturbed.split(ID_BATCH):
                    scores.append(model(batch).softmax(1)[:,target].double().cpu().numpy())
                curve=np.concatenate(scores).reshape(2,ID_STEPS+1)
                assert np.isfinite(curve).all() and curve.min()>=0 and curve.max()<=1+1e-6
                assert np.allclose(curve[:,[0,-1]],[[baseline_prob[image_index],full_prob],
                                    [full_prob,baseline_prob[image_index]]],atol=3e-5,rtol=1e-4)
                curves[image_index,method_index]=curve
                del masks,perturbed
            if (image_index+1)%10==0: print(name,'I/D curves',image_index+1,'/',ID_N,flush=True)
    all_zero=(cams.reshape(ID_N,7,-1).max(2)==0)
    for image_index,slot in np.argwhere(all_zero):
        assert np.allclose(curves[image_index,slot],curves[image_index,7],atol=3e-5,rtol=1e-4)
    arrays={'image_ids':np.array(ID_IMAGE_IDS),'cams':cams,'targets':targets,
        'original_prob':original_prob,'baseline_prob':baseline_prob,'curves':curves,'all_zero':all_zero}
    np.savez_compressed(cache,**arrays)
    details={'signature':signature,'sha256':sha256_file(cache),'seconds':time.monotonic()-started,
        'gradcam_seconds':grad_seconds,'relu_forward_equivalence_max_error':equivalence_error,
        'backward_warnings':sorted(warnings_seen),'all_zero_maps':int(all_zero.sum()),
        'endpoint_checks_passed':True,'zero_map_random_equivalence_passed':True}
    meta.write_text(json.dumps(details,indent=2),encoding='utf-8')
    id_data[name]=arrays
    print('Completed I/D:',name,'seconds',round(details['seconds'],1),'zero maps',int(all_zero.sum()),flush=True)
    del model,curves,cams
    gc.collect()
    if DEVICE.type=='cuda': torch.cuda.empty_cache()
'''))
new.append(cell('markdown',r'''
## 10. Insertion/Deletion结果、随机对照与不确定性

下表包含三个模型各七个取样点及随机对照的平均AUC。
配对差的区间以图片为抽样单位；每次抽到某图时，同时取它的CAM和随机对照值。
该区间仅描述这100个探索性样本的抽样变化，不覆盖基线选择、梯度随机性或ImageNet全体代表性的全部不确定性。

曲线是100张图的平均目标概率。即便某层没有优于随机、出现全零图或曲线并不单调，也按实际结果保留。
不要用一个AUC方向把Insertion和Deletion混在一起比较；两个指标可能给出不同排序。
'''))
new.append(cell('code',r'''
auc_rows=[]; curve_rows=[]; id_summary_rows=[]
bootstrap_indices=np.random.default_rng(SEED+30000).integers(0,ID_N,size=(ID_REPEATS,ID_N))
for name,arrays in id_data.items():
    aucs=auc_probability(arrays['curves'])
    assert aucs.shape==(ID_N,8,2)
    for method_index,method in enumerate(ID_METHODS):
        ins=aucs[:,method_index,0]; deletion=aucs[:,method_index,1]
        ins_gain=ins-aucs[:,7,0]; del_gain=aucs[:,7,1]-deletion
        ins_ci=np.quantile(ins_gain[bootstrap_indices].mean(1),[.025,.975])
        del_ci=np.quantile(del_gain[bootstrap_indices].mean(1),[.025,.975])
        id_summary_rows.append({'model':name,'method':method,'n':ID_N,
            'insertion_auc_mean':float(ins.mean()),'deletion_auc_mean':float(deletion.mean()),
            'insertion_gain_vs_random':float(ins_gain.mean()),'ins_gain_ci_low':float(ins_ci[0]),'ins_gain_ci_high':float(ins_ci[1]),
            'deletion_gain_vs_random':float(del_gain.mean()),'del_gain_ci_low':float(del_ci[0]),'del_gain_ci_high':float(del_ci[1]),
            'all_zero_maps':int(arrays['all_zero'][:,method_index].sum()) if method_index<7 else 0})
        for image_index,image_id in enumerate(ID_IMAGE_IDS):
            auc_rows.append({'model':name,'image_id':image_id,'target_class':int(arrays['targets'][image_index]),
                'method':method,'insertion_auc':float(ins[image_index]),'deletion_auc':float(deletion[image_index]),
                'insertion_gain_vs_random':float(ins_gain[image_index]),'deletion_gain_vs_random':float(del_gain[image_index])})
            for metric_index,metric in enumerate(['insertion','deletion']):
                for step,fraction in enumerate(ID_FRACTIONS):
                    curve_rows.append({'model':name,'image_id':image_id,'method':method,'metric':metric,
                        'fraction':float(fraction),'target_probability':float(arrays['curves'][image_index,method_index,metric_index,step])})
id_summary=pd.DataFrame(id_summary_rows)
id_summary.to_csv(OUT/'results'/'id_summary.csv',index=False)
pd.DataFrame(auc_rows).to_csv(OUT/'results'/'id_per_image_auc.csv',index=False)
pd.DataFrame(curve_rows).to_csv(OUT/'results'/'id_curves.csv',index=False)
display(id_summary.round(5))
fig,axes=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
colors=plt.cm.tab10(np.arange(7))
for row,(name,arrays) in enumerate(id_data.items()):
    for metric_index,metric in enumerate(['Insertion (higher AUC better)','Deletion (lower AUC better)']):
        ax=axes[row,metric_index]
        for k,method in enumerate(ID_METHODS):
            mean_curve=arrays['curves'][:,k,metric_index].mean(0)
            ax.plot(ID_FRACTIONS,mean_curve,label=method,color=colors[k] if k<7 else 'black',
                    linestyle='-' if k<7 else '--',linewidth=1.5)
        ax.set(title=f'{name}: {metric}',xlabel='Fraction of pixels modified',
               ylabel='Fixed-target probability (mean of 100 images)',xlim=(0,1),ylim=(0,1))
        ax.grid(alpha=.2)
        if row==0: ax.legend(ncol=4,fontsize=8)
fig.savefig(OUT/'figures'/'insertion_deletion_curves.png',dpi=170)
display(fig);plt.close(fig)
'''))
new.append(cell('code',r'''
# One fixed, unfiltered example; no selection for visual appeal.
fig,axes=plt.subplots(3,5,figsize=(18,10),constrained_layout=True)
for row,(name,arrays) in enumerate(id_data.items()):
    x=dataset[0][0]
    rgb=(x.numpy().transpose(1,2,0)*np.array([.229,.224,.225])+np.array([.485,.456,.406])).clip(0,1)
    cam=arrays['cams'][0,6]
    colored=plt.cm.inferno(cam)[...,:3]
    overlay=rgb*(1-.5*cam[...,None])+colored*(.5*cam[...,None])
    order=ranked_pixels(cam,0)
    rank=np.empty(PIXELS,dtype=np.int64);rank[order]=np.arange(PIXELS)
    mask=(rank<ID_COUNTS[5]).reshape(224,224,1)
    baseline=np.broadcast_to(np.array([.485,.456,.406]),rgb.shape)
    views=[rgb,overlay,np.where(mask,rgb,baseline),np.where(mask,baseline,rgb)]
    target=int(arrays['targets'][0]); category=SPECS[name][1].meta['categories'][target]
    titles=[f'{name}\nTarget: {category}',f'L07 Grad-CAM; zero={bool(arrays["all_zero"][0,6])}',
            'Insertion: 25% restored','Deletion: 25% replaced']
    for ax,view,title in zip(axes[row,:4],views,titles): ax.imshow(view);ax.set_title(title,fontsize=10);ax.axis('off')
    ax=axes[row,4]
    ax.plot(ID_FRACTIONS,arrays['curves'][0,6,0],label='Insertion')
    ax.plot(ID_FRACTIONS,arrays['curves'][0,6,1],label='Deletion')
    ax.set(title='L07 fixed-target curves',xlabel='Fraction',ylabel='Probability',ylim=(0,1))
    ax.legend(fontsize=8);ax.grid(alpha=.2)
fig.suptitle(f'Fixed example: {ID_IMAGE_IDS[0]} — no cross-map absolute color comparison',fontsize=13)
fig.savefig(OUT/'figures'/'insertion_deletion_example.png',dpi=170)
display(fig);plt.close(fig)
'''))
new.append(cell('markdown',r'''
## 11. 新增结果解读与CKA的关系

下面根据实际AUC和配对区间生成结论。跨21个层寻找“最佳”属于事后探索，不能当作预先指定、经多重比较校正的发现。
不同模型可能解释不同Top-1类别，所以模型间AUC的差别不等价于同一类别上的解释质量或分类能力差别。

CKA高的层可以有较差的Insertion/Deletion表现：前者比较跨图片的表示几何，后者检查特定类别的空间排序。
例如浅层GAP可能保留相似的全局颜色统计，但这种表示相似性本身不保证Grad-CAM能精确定位目标区域。
'''))
new.append(cell('code',r'''
id_lines=['# Grad-CAM Insertion/Deletion：实际结果与解释',
    '固定ImageNet子集前100张、不按正确率筛选；每模型每图固定原始Top-1，七层共享目标。',
    '基线为ImageNet均值颜色；21个采样点，概率AUC；Insertion高为好，Deletion低为好。']
for name in SPECS:
    table=id_summary[(id_summary['model']==name)&(id_summary['method']!='random')]
    random_row=id_summary[(id_summary['model']==name)&(id_summary['method']=='random')].iloc[0]
    best_ins=table.loc[table['insertion_auc_mean'].idxmax()]
    best_del=table.loc[table['deletion_auc_mean'].idxmin()]
    wins_ins=int((table['ins_gain_ci_low']>0).sum())
    wins_del=int((table['del_gain_ci_low']>0).sum())
    id_lines.extend([f'\n## {name}',
        f'随机排序平均Insertion／Deletion AUC：{random_row.insertion_auc_mean:.4f}／{random_row.deletion_auc_mean:.4f}。',
        f'本样本中Insertion最高为{best_ins.method}，AUC={best_ins.insertion_auc_mean:.4f}；Deletion最低为{best_del.method}，AUC={best_del.deletion_auc_mean:.4f}。',
        f'在Insertion／Deletion上，分别有{wins_ins}／{wins_del}个层的配对改进95%区间完全位于0以上；这是未经多重比较校正的探索性描述。',
        f'七层中共{int(table.all_zero_maps.sum())}张全零CAM；它们保留在评估中，并已检查与随机排序曲线一致。'])
id_lines.extend(['\n## 怎样解释，而不是过度解读',
    '1. CAM相对随机排序的正向改进，表示其在本协议下更快恢复／破坏该固定目标的分数；不等于分类准确率提高。',
    '2. 若Insertion和Deletion选择不同层，应分别报告，不能只挑对自己有利的一个指标。',
    '3. 固定目标、原始／基线端点、像素计数、AUC方向、非inplace前向等价及全零图行为均在代码中校验。',
    '4. 只用均值颜色基线、100个类别各一张、20个区间及一个随机排序／图；未覆盖其他基线、步长与所有可能随机排序。bootstrap区间不覆盖全部这些不确定性。',
    '5. 遮挡图片可能偏离自然图像分布；概率受模型校准影响；不同模型Top-1可能不同，所以不将这些AUC用作模型准确率排行榜。',
    '6. CKA与扰动评估互补：表示相似并不保证解释图相同或扰动评估一样好。本实验不把两者的相关性当作已验证因果关系。'])
id_text='\n\n'.join(id_lines)
display(Markdown(id_text))
(OUT/'insertion_deletion_summary.md').write_text(id_text,encoding='utf-8')
id_config={'protocol':ID_PROTOCOL,'complete':True,'n_models':3,'cam_combinations':2100,
    'random_controls':300,'summary_sha256':sha256_file(OUT/'results'/'id_summary.csv'),
    'per_image_auc_sha256':sha256_file(OUT/'results'/'id_per_image_auc.csv'),
    'curves_sha256':sha256_file(OUT/'results'/'id_curves.csv'),
    'cache_sha256':{name:sha256_file(ID_DIR/f'{name}_id.npz') for name in SPECS}}
(ID_DIR/'experiment_config.json').write_text(json.dumps(id_config,indent=2),encoding='utf-8')
config['insertion_deletion']=id_config
(OUT/'experiment_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print('Completed: CKA + 2100 Grad-CAM I/D evaluations + 300 random controls.')
'''))

# Preserve the original executed cells; add the new self-contained experiment afterwards.
nb['cells'][0]['source']=nb['cells'][0]['source'].replace('Linear CKA 研究','Linear CKA 与 Insertion/Deletion 研究')
nb['cells'][0]['source']+='\n新增：对100张固定图片、三个模型各七层的Grad-CAM进行Insertion/Deletion评估、随机对照及配对bootstrap分析。\n'
nb['cells'][17]['source']=nb['cells'][17]['source'].replace('## 9.','## 12.')
nb['cells'][17]['source']+='\n新增参考：[Grad-CAM](https://arxiv.org/abs/1610.02391)、[RISE扰动评估](https://arxiv.org/abs/1806.07421)。Insertion/Deletion基线与AUC方向详见新增章节。\n'
# Resource lookup works from the root notebook, submit/, or a fresh standalone folder.
s=nb['cells'][2]['source']
s=s.replace("ROOT = Path.cwd().resolve()", "ROOT = Path.cwd().resolve()\nif not (ROOT/'cka_study'/'imagenet_subset'/'manifest.json').exists() and (ROOT.parent/'cka_study'/'imagenet_subset'/'manifest.json').exists():\n    ROOT = ROOT.parent")
nb['cells'][2]['source']=s
nb['cells']=nb['cells'][:17]+[nb['cells'][18]]+new+[nb['cells'][17]]
nb['metadata']['task_status']='CKA complete; Insertion/Deletion section pending execution'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Extended notebook:',len(nb['cells']),'cells')
