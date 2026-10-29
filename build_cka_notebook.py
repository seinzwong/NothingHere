"""Build the single, self-contained deliverable notebook using only stdlib."""
from pathlib import Path
import ast
import json
import textwrap
import uuid

ROOT = Path(__file__).resolve().parent
cells = []


def md(source):
    cells.append(dict(cell_type='markdown', id=uuid.uuid4().hex[:8], metadata={},
                      source=textwrap.dedent(source).strip() + '\n'))


def code(source):
    source = textwrap.dedent(source).strip() + '\n'
    ast.parse(source)
    cells.append(dict(cell_type='code', id=uuid.uuid4().hex[:8], metadata={},
                      source=source, execution_count=None, outputs=[]))


md(r'''
# 三种 CNN 的中间表示像不像？ImageNet 上的 Linear CKA 研究

**我的贡献：**对 VGG16、ResNet50、MobileNetV2 的七个卷积位置进行跨模型定量比较。
沿用团队的 ImageNet-1K V1 权重和搭档1的取样点，补充 CKA 证据；不重训模型或探针，不重复搭档的 Lucent 图像合成。

本 notebook 包含环境准备、官方下载与校验、真实特征采集、数值检查、147个主要分数、抽样及错配对照、图表和结果驱动的解释。
**正式数据固定为 ImageNet validation，每类一张，共1,000张；不会自动改用 CIFAR10。**

使用方法：在 Jupyter／VS Code 选择 `C:\Anaconda3\envs\5070ti\python.exe`，从上到下运行。
首次执行需要网络；ImageNet 需要在本机登录 Hugging Face 并接受数据访问条件。不要把 token 写入代码或输出。
资源与特征缓存在本项目中，后续运行优先验证并复用。首次下载可能较久；若中断可重新运行。

研究问题：相同图片经过不同架构后，浅、中、深采样位置的表示相似度如何变化？最相似的层是否出现在相同槽位？
预期“浅层相似、深层不同”只是待检验假设，下面不预填结论。

**与搭档的关系：**搭档2 `A2-models.ipynb` 主要合成通道偏好图；搭档1的两张图研究包含特征图、Grad-CAM、线性探针。
我为 CKA 采集真实激活，比较样本间关系，三个模块互为补充。
''')

md(r'''
## 1. 环境准备与可复现设置

架构是计算结构，权重是学到的参数；预训练表示我们直接使用已训练的模型。
`IMAGENET1K_V1` 是权重版本名，不是本次下载图片的文件名。GPU 负责前向计算，CPU 用 float64 计算 CKA。

当前目标环境：Python 3.10、torch 2.9.1+cu130、torchvision 0.24.1+cu130。
代码先检查依赖；默认不更改现有环境。只有自己新建环境且确实缺包时，才开启安装开关。
本机已注册 `Python (5070ti)` 内核；在其他机器配置后，可用目标环境的Python执行 `python -m ipykernel install --user --name 5070ti --display-name "Python (5070ti)"` 注册。
`eval()` 控制 BN／Dropout 的评估行为，冻结参数表示不训练，`no_grad()` 避免建立梯度图，三者作用不同。
''')
code(r'''
import os, sys, subprocess, importlib.util, platform, json, hashlib, time, random, gc
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_CACHE'] = str(Path.cwd()/'.cache'/'huggingface'/'hub')
os.environ['HF_DATASETS_CACHE'] = str(Path.cwd()/'.cache'/'huggingface'/'datasets')
INSTALL_MISSING = False
packages = {'torch':'torch==2.9.1', 'torchvision':'torchvision==0.24.1',
            'numpy':'numpy', 'pandas':'pandas', 'PIL':'Pillow', 'matplotlib':'matplotlib',
            'datasets':'datasets', 'huggingface_hub':'huggingface_hub', 'psutil':'psutil'}
missing = [package for module, package in packages.items() if importlib.util.find_spec(module) is None]
if missing and not INSTALL_MISSING:
    raise RuntimeError(f'当前解释器 {sys.executable} 缺少 {missing}；请选择5070ti环境，或在新环境启用安装开关。')
if missing:
    gpu_packages = [x for x in missing if x.startswith(('torch==','torchvision=='))]
    other_packages = [x for x in missing if x not in gpu_packages]
    if gpu_packages:
        subprocess.check_call([sys.executable,'-m','pip','install',*gpu_packages,
                               '--index-url','https://download.pytorch.org/whl/cu130'])
    if other_packages:
        subprocess.check_call([sys.executable,'-m','pip','install',*other_packages])
    importlib.invalidate_caches()

import numpy as np, pandas as pd, matplotlib.pyplot as plt, psutil
import torch, torchvision, datasets, huggingface_hub
from torchvision import models, transforms
from PIL import Image
from IPython.display import display, Markdown, HTML
from huggingface_hub import HfApi, get_token, hf_hub_url, get_hf_file_metadata
get_ipython().run_line_magic('matplotlib', 'inline')

ROOT = Path.cwd().resolve()
print('项目输出根目录：',ROOT)
OUT = ROOT/'cka_study'
for name in ['features','results','figures','imagenet_subset']:
    (OUT/name).mkdir(parents=True, exist_ok=True)
torch.hub.set_dir(str(ROOT/'.cache'/'torch'/'hub'))
SEED = 5495
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    z = torch.randn(128,128,device=DEVICE)
    assert torch.isfinite(z@z.T).all().item()
    del z
environment = {'python':platform.python_version(), 'executable':sys.executable,
    'torch':torch.__version__, 'torchvision':torchvision.__version__,
    'datasets':datasets.__version__, 'huggingface_hub':huggingface_hub.__version__,
    'numpy':np.__version__, 'cuda_runtime':torch.version.cuda, 'device':str(DEVICE),
    'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    'gpu_vram_gib':torch.cuda.get_device_properties(0).total_memory/1024**3 if torch.cuda.is_available() else None,
    'cpu_physical_cores':psutil.cpu_count(logical=False), 'cpu_logical_cores':psutil.cpu_count(),
    'ram_gib':psutil.virtual_memory().total/1024**3}
display(pd.DataFrame([environment]).T.rename(columns={0:'实际环境'}))
print('仅下载与前向；不训练CNN，不需要Lucent或Captum。')
''')

md(r'''
## 2. 模型、取样点和预处理

Tap 是取出数值的位置；hook 是运行到该模块时记录输出的机制。
下面所有点都是 **Conv2d 原始输出，在后续 BN／ReLU 之前**。BN 是归一化层，ReLU 把负数置零。
ResNet conv3 输出也在残差相加前，不等于整块输出。L01–L07按主路径卷积序号近似等距，不是语义一一对应。
搭档2的五层配置包含 BN 和整个block，与这套配置不同，不能混用名称或缓存。

预处理统一为RGB、短边256、中心裁剪224、转0～1、ImageNet RGB Normalize。
这与搭档1两张照片的letterbox补边不同，应分别解释。
GAP（全局平均池化）将每通道的空间位置求平均：`[B,C,H,W] → [B,C]`。
一行是一张图，列数是通道数，可跨模型不同。GAP丢失空间排列，本文比较的是GAP表示。
''')
code(r'''
SPECS = {
 'vgg16':(models.vgg16,models.VGG16_Weights.IMAGENET1K_V1,
          ['features.0','features.5','features.10','features.14','features.19','features.24','features.28']),
 'resnet50':(models.resnet50,models.ResNet50_Weights.IMAGENET1K_V1,
          ['conv1','layer1.2.conv2','layer2.2.conv1','layer3.0.conv3','layer3.3.conv2','layer4.0.conv1','layer4.2.conv3']),
 'mobilenet_v2':(models.mobilenet_v2,models.MobileNet_V2_Weights.IMAGENET1K_V1,
          ['features.0.0','features.4.conv.0.0','features.6.conv.2','features.9.conv.2',
           'features.12.conv.1.0','features.15.conv.1.0','features.18.0'])}
PREPROCESS = models.VGG16_Weights.IMAGENET1K_V1.transforms()
preprocess_config = str(PREPROCESS)
for _,weights,_ in SPECS.values():
    assert str(weights.transforms()) == preprocess_config
display(pd.DataFrame({name:spec[2] for name,spec in SPECS.items()},index=[f'L{i:02d}' for i in range(1,8)]))
print(PREPROCESS)

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

model_resources={}
for name,(builder,weights,layers) in SPECS.items():
    path=Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[1]
    expected=path.stem.rsplit('-',1)[1]
    if not path.exists():
        # 官方API校验文件名给定的SHA256前缀；已缓存则不重复下载。
        weights.get_state_dict(progress=False,check_hash=True)
    actual=sha256_file(path)
    assert actual.startswith(expected), f'{name}权重校验失败，请隔离损坏缓存后重新下载。'
    model_resources[name]={'weights':str(weights),'url':weights.url,'sha256':actual,
                            'bytes':path.stat().st_size,'layers':layers}
    print(name,weights.name,'SHA256:',actual)
''')

md(r'''
## 3. ImageNet访问和子集获取

**只用官方 `ILSVRC/imagenet-1k` 的 validation，不用train或无标签test。**
首次连接时记录仓库commit，以后沿用固定版本。访问前请在数据页接受条款，并在本机执行：

```powershell
& 'C:\Anaconda3\envs\5070ti\Scripts\hf.exe' auth login
```

这里使用本机凭据，代码不显示或保存token。若没有授权，本节明确停止；后续不生成虚构结果。
采用流式读取，按该版本的确定顺序保留每类第一张可解码、字节不重复的图片，直到覆盖1,000类；最后用固定种子打乱。
**这是每类配额抽样，不是全验证集均匀随机抽样**。扫描可能需要读取额外图片，网络流量大于最终保存的1,000张。
哈希识别完全相同的字节，不能排除所有近似重复图。
''')
code(r'''
REPO='ILSVRC/imagenet-1k'
data_manifest_path=OUT/'imagenet_subset'/'manifest.json'
revision_path=OUT/'imagenet_subset'/'revision.json'

def load_or_download_imagenet():
    if data_manifest_path.exists():
        m=json.loads(data_manifest_path.read_text(encoding='utf-8'))
        assert m['repo']==REPO and m['split']=='validation' and len(m['images'])==1000
        assert sorted(r['label'] for r in m['images'])==list(range(1000))
        assert len({r['image_id'] for r in m['images']})==1000
        for r in m['images']:
            assert sha256_file(ROOT/r['path'])==r['sha256'], r['path']
        print('复用已校验的ImageNet子集：1000张，1000类。')
        return m
    if not get_token():
        raise RuntimeError('需要本机HF登录且接受ImageNet访问条款；不会切换为CIFAR10。')
    api=HfApi()
    if revision_path.exists():
        revision=json.loads(revision_path.read_text(encoding='utf-8'))['revision']
    else:
        revision=api.dataset_info(REPO,token=True).sha
        revision_path.write_text(json.dumps({'repo':REPO,'revision':revision}),encoding='utf-8')
    files=api.list_repo_files(REPO,repo_type='dataset',revision=revision,token=True)
    valfiles=[x for x in files if 'validation' in x.lower() and x.endswith('.parquet')]
    if not valfiles:
        raise RuntimeError('固定版本未找到validation parquet，需核对仓库结构，不自动换数据。')
    # 只验证读取权限，不将带凭据的URL写入输出。
    get_hf_file_metadata(hf_hub_url(REPO,valfiles[0],repo_type='dataset',revision=revision),token=True)
    stream=datasets.load_dataset(REPO,split='validation',revision=revision,
        streaming=True,token=True,cache_dir=str(OUT/'hf_cache'))
    stream=stream.cast_column('image',datasets.Image(decode=False))
    import io
    chosen={}; seen_hashes=set(); invalid=0; scanned=0
    started=time.monotonic()
    for row_index,row in enumerate(stream):
        scanned=row_index+1
        label=int(row['label'])
        if label in chosen: continue
        value=row['image']; raw=value.get('bytes')
        if raw is None:
            raise RuntimeError('流式记录没有图片字节，请核对数据格式。')
        digest=hashlib.sha256(raw).hexdigest()
        if digest in seen_hashes: continue
        try:
            with Image.open(io.BytesIO(raw)) as im:
                im.convert('RGB').load()
                width,height=im.size
        except (OSError,ValueError):
            invalid+=1; continue
        path=OUT/'imagenet_subset'/f'class_{label:04d}_{digest[:16]}.img'
        path.write_bytes(raw)
        chosen[label]={'image_id':f'imagenet-validation-{row_index:05d}', 'row_index':row_index,
            'label':label, 'sha256':digest,'path':str(path.relative_to(ROOT)),
            'width':width,'height':height}
        seen_hashes.add(digest)
        if len(chosen)%100==0:
            print(f'已取得{len(chosen)}/1000类，扫描{scanned}张，耗时{time.monotonic()-started:.1f}秒',flush=True)
        if len(chosen)==1000: break
    if len(chosen)!=1000:
        raise RuntimeError(f'仅取得{len(chosen)}类；不把不完整子集当成正式结果。')
    records=[chosen[c] for c in sorted(chosen)]
    np.random.default_rng(SEED).shuffle(records)
    m={'repo':REPO,'revision':revision,'split':'validation','selection':'first eligible unique bytes per class in stream order',
       'seed_for_final_order':SEED,'n':1000,'classes':1000,'scanned_rows':scanned,
       'invalid_images_skipped':invalid,'images':records}
    data_manifest_path.write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    return m

data_manifest=load_or_download_imagenet()
IMAGE_IDS=[r['image_id'] for r in data_manifest['images']]
print('数据版本:',data_manifest['revision'],'划分:',data_manifest['split'])
print('每类1张，共',len(IMAGE_IDS),'张；扫描记录数:',data_manifest['scanned_rows'])
fig,axes=plt.subplots(2,5,figsize=(12,5))
for ax,r in zip(axes.flat,data_manifest['images'][:10]):
    with Image.open(ROOT/r['path']) as im: ax.imshow(im.convert('RGB'))
    ax.set_title(f"class {r['label']}"); ax.axis('off')
fig.suptitle('ImageNet validation examples (before preprocessing)')
display(fig); plt.close(fig)
''')

md(r'''
## 4. 提取真实中间激活并缓存

inplace是原地修改；`detach().clone()` 立即复制卷积输出，防止后续激活改写。
每个模型每个batch只前向一次，同时取得七层。保留全部通道，不使用展示用的16个通道，也不使用探针标准化或分类概率。
缓存与数据清单、模型哈希、层名、预处理和软件版本绑定；不能只因为文件存在就盲目复用。
下面先做一个batch检查，再完成全部图片。模型逐个加载，减少16GB显存压力。
''')
code(r'''
class ImageNetSubset(torch.utils.data.Dataset):
    def __len__(self): return len(data_manifest['images'])
    def __getitem__(self,index):
        r=data_manifest['images'][index]
        with Image.open(ROOT/r['path']) as im: x=PREPROCESS(im.convert('RGB'))
        return x,index

dataset=ImageNetSubset()
BATCH_SIZE=8
features={}; extraction_stats={}
for name,(builder,weights,layers) in SPECS.items():
    signature={'model':model_resources[name], 'data_sha256':sha256_file(data_manifest_path),
        'preprocess':preprocess_config,'representation':'raw Conv2d output before following BN/ReLU; all-channel GAP; no feature standardization',
        'torch':torch.__version__,'torchvision':torchvision.__version__,'deterministic':True,'tf32':False}
    cache=OUT/'features'/f'{name}_gap.npz'
    meta=cache.with_suffix('.json')
    if cache.exists() and meta.exists():
        saved=json.loads(meta.read_text(encoding='utf-8'))
        if saved['signature']==signature and saved['sha256']==sha256_file(cache):
            with np.load(cache,allow_pickle=False) as z:
                assert z['image_ids'].tolist()==IMAGE_IDS
                features[name]=[z[f'layer{i:02d}'].copy() for i in range(1,8)]
            extraction_stats[name]=saved['extraction']
            print('复用合格特征缓存:',name)
            continue
    model=builder(weights=weights,progress=False).eval().to(DEVICE)
    for p in model.parameters(): p.requires_grad_(False)
    modules=dict(model.named_modules())
    assert all(isinstance(modules[layer],torch.nn.Conv2d) for layer in layers)
    current={}; handles=[]; shapes={}
    def make_hook(slot):
        def hook(module,inputs,output):
            assert output.ndim==4
            shapes[slot]=list(output.shape[1:])
            current[slot]=output.detach().clone().mean(dim=(2,3)).cpu().numpy()
        return hook
    for slot,layer in enumerate(layers): handles.append(modules[layer].register_forward_hook(make_hook(slot)))
    parts=[[] for _ in layers]; indices=[]; started=time.monotonic(); batches=0
    if DEVICE.type=='cuda': torch.cuda.reset_peak_memory_stats()
    try:
        loader=torch.utils.data.DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=False,
                    num_workers=0,pin_memory=DEVICE.type=='cuda')
        with torch.no_grad():
            for images,idx in loader:
                current.clear()
                output=model(images.to(DEVICE,non_blocking=True))
                assert set(current)==set(range(7))
                assert output.shape[1]==1000 and torch.isfinite(output).all().item()
                for slot in range(7): parts[slot].append(current[slot])
                indices.extend(idx.tolist()); batches+=1
                if batches==1 or batches%25==0:
                    print(name,'processed',len(indices),'/',len(dataset),flush=True)
        assert indices==list(range(len(dataset)))
        features[name]=[np.concatenate(p,axis=0).astype(np.float32,copy=False) for p in parts]
        assert all(x.shape[0]==1000 and np.isfinite(x).all() for x in features[name])
        np.savez_compressed(cache,image_ids=np.array(IMAGE_IDS),**{f'layer{i+1:02d}':x for i,x in enumerate(features[name])})
        stats={'device':str(DEVICE),'seconds':time.monotonic()-started,'batches':batches,
               'batch_size':BATCH_SIZE,'raw_shapes':{f'L{i+1:02d}':shapes[i] for i in range(7)},
               'peak_allocated_vram_gib':torch.cuda.max_memory_allocated()/1024**3 if DEVICE.type=='cuda' else None}
        extraction_stats[name]=stats
        meta.write_text(json.dumps({'signature':signature,'sha256':sha256_file(cache),'extraction':stats},indent=2),encoding='utf-8')
        print('完成:',name,'seconds:',round(stats['seconds'],1))
    finally:
        for handle in handles: handle.remove()
        del model
        current.clear(); gc.collect()
        if DEVICE.type=='cuda': torch.cuda.empty_cache()

shape_rows=[{'model':name,'slot':f'L{i+1:02d}','module':SPECS[name][2][i],
             'N':x.shape[0],'C':x.shape[1]} for name,xs in features.items() for i,x in enumerate(xs)]
display(pd.DataFrame(shape_rows))
display(pd.DataFrame(extraction_stats).T[['device','seconds','batches','peak_allocated_vram_gib']])
''')

md(r'''
## 5. Linear CKA公式与数值自检

中心化：每个特征列减去其在所有图片上的平均值，得到Xc、Yc。

\[
\mathrm{CKA}(X,Y)=\frac{\|X_c^T Y_c\|_F^2}{\|X_c^T X_c\|_F\,\|Y_c^T Y_c\|_F}.
\]

T是转置；F范数是矩阵元素平方和开平方。
也可比较样本内积矩阵（Gram矩阵）`K=Xc@Xc.T`、`L=Yc@Yc.T`，两种实现等价。
普通、非去偏Linear CKA在非退化时介于0和1；不是准确率，没有通用“0.8即好”的标准。
只有一张图时中心化全零；两张图在非退化时恒为1，因此本研究使用多图片。
逐通道标准化会改变表示，不能与CKA中心化混为一谈。
''')
code(r'''
def center_features(x):
    x=np.asarray(x,dtype=np.float64)
    if x.ndim!=2 or x.shape[0]<3 or not np.isfinite(x).all():
        raise ValueError('需要至少3行的有限二维矩阵；正式实验使用1000张。')
    return x-x.mean(axis=0,keepdims=True)

def normalized_gram(x):
    x=center_features(x)
    g=x@x.T
    norm=np.linalg.norm(g,'fro')
    if norm==0 or not np.isfinite(norm): raise ValueError('零变化表示或无效范数，CKA未定义。')
    return g/norm

def linear_cka(x,y):
    if len(x)!=len(y): raise ValueError('样本行数不同。')
    return float(np.sum(normalized_gram(x)*normalized_gram(y)))

def feature_space_cka(x,y):
    x,y=center_features(x),center_features(y)
    denom=np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro')
    if denom==0: raise ValueError('CKA未定义。')
    return float(np.linalg.norm(x.T@y,'fro')**2/denom)

rng=np.random.default_rng(SEED)
x,y=rng.normal(size=(48,12)),rng.normal(size=(48,9))
p=rng.permutation(48); base=linear_cka(x,y)
checks={'self':np.isclose(linear_cka(x,x),1),
        'symmetry':np.isclose(base,linear_cka(y,x)),
        'joint_permutation':np.isclose(base,linear_cka(x[p],y[p])),
        'translation':np.isclose(base,linear_cka(x+np.arange(12),y+2)),
        'uniform_scaling':np.isclose(base,linear_cka(-3*x,2*y)),
        'two_equivalent_implementations':np.isclose(base,feature_space_cka(x,y),atol=1e-10)}
for label,a,b in [('constant',np.ones((48,3)),y),('two_samples',x[:2],y[:2]),
                  ('different_N',x,y[:-1]),('nonfinite',np.full((48,3),np.nan),y)]:
    try: linear_cka(a,b); checks['reject_'+label]=False
    except ValueError: checks['reject_'+label]=True
assert all(checks.values()),checks
display(pd.Series(checks,name='passed').to_frame())
''')

md(r'''
## 6. 正式结果：三个模型配对，每对49个层组合

每个模型七层，三对模型共 `3×7×7=147` 个分数。
不只看同槽位，避免预先假设不同架构的层语义对应。矩阵图是CKA数值的展示，非Grad-CAM空间归因。
''')
code(r'''
names=list(SPECS)
PAIRS=[(names[0],names[1]),(names[0],names[2]),(names[1],names[2])]
grams={name:np.stack([normalized_gram(x) for x in features[name]]) for name in names}
flat={name:g.reshape(7,-1) for name,g in grams.items()}
matrices={(a,b):flat[a]@flat[b].T for a,b in PAIRS}
assert all(np.isfinite(m).all() and m.min()>=-1e-9 and m.max()<=1+1e-9 for m in matrices.values())
audit_rows=[]
for (a,b),m in matrices.items():
    for i in [0,3,6]:
        independent=feature_space_cka(features[a][i],features[b][i])
        error=abs(independent-m[i,i])
        assert error<1e-10
        audit_rows.append({'model_a':a,'model_b':b,'slot':i+1,
                           'gram_cka':float(m[i,i]),'feature_space_cka':independent,'absolute_error':error})
numeric_audit=pd.DataFrame(audit_rows)
numeric_audit.to_csv(OUT/'results'/'cka_numeric_audit.csv',index=False)
print('9个真实浅／中／深组合经独立特征空间公式复算通过；最大误差：',numeric_audit['absolute_error'].max())
rows=[]
for (a,b),m in matrices.items():
    for i in range(7):
        for j in range(7):
            rows.append({'model_a':a,'slot_a':f'L{i+1:02d}','module_a':SPECS[a][2][i],
                'model_b':b,'slot_b':f'L{j+1:02d}','module_b':SPECS[b][2][j],
                'cka':float(m[i,j]),'n':1000,'dataset':'ImageNet validation',
                'representation':'raw_conv_GAP','estimator':'ordinary_linear_CKA'})
results=pd.DataFrame(rows)
assert len(results)==147
results.to_csv(OUT/'results'/'cka_main.csv',index=False)
display(HTML('<details open><summary>完整147个CKA数值（可折叠）</summary>'+results.to_html(index=False,float_format=lambda x:f'{x:.6f}')+'</details>'))
fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
for ax,((a,b),m) in zip(axes,matrices.items()):
    im=ax.imshow(m,vmin=0,vmax=1,cmap='viridis')
    ax.set_title(f'{a} vs {b}')
    ax.set_xlabel(b); ax.set_ylabel(a)
    ax.set_xticks(range(7),[f'L{i}' for i in range(1,8)])
    ax.set_yticks(range(7),[f'L{i}' for i in range(1,8)])
    for i in range(7):
        for j in range(7): ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=8,
                                color='white' if m[i,j]<0.5 else 'black')
fig.colorbar(im,ax=axes,label='Ordinary Linear CKA (raw Conv2d GAP)')
fig.savefig(OUT/'figures'/'cka_matrices.png',dpi=180)
display(fig); plt.close(fig)
''')

md(r'''
## 7. 可靠性：样本规模和错配对照

普通CKA可能受样本量及维度影响。只打乱一侧图片ID，使正确配对关系被破坏，作为诊断对照，不能预设它一定接近0。
重复10次错配，保存每个层组合的结果；不称为统计显著性检验。
250、500、1000张是同一主集的嵌套子集；由于每类仅一张，规模变化也改变类别覆盖，不能将差异全归因于N。
从全样本Gram截取子矩阵后必须重新中心化；直接截取计算会错误。
''')
code(r'''
def subset_gram(g,idx):
    s=g[np.ix_(idx,idx)].copy()
    s=s-s.mean(axis=0,keepdims=True)-s.mean(axis=1,keepdims=True)+s.mean()
    norm=np.linalg.norm(s,'fro')
    if norm==0: raise ValueError('子集表示退化。')
    return s/norm

check_rows=[]; stability={}; shuffle_matrices={pair:[] for pair in PAIRS}
for n in [250,500,1000]:
    idx=np.arange(n)
    fs={name:np.stack([subset_gram(g,idx).ravel() for g in grams[name]]) for name in names}
    for a,b in PAIRS:
        m=fs[a]@fs[b].T
        stability[(a,b,n)]=m
        for i in range(7):
            for j in range(7): check_rows.append({'kind':'sample_size','model_a':a,'model_b':b,
                'slot_a':i+1,'slot_b':j+1,'n':n,'repeat':0,'seed':SEED,'cka':float(m[i,j])})
    del fs
# 验证子集Gram重中心化与直接重算一致。
assert np.allclose(subset_gram(grams[names[0]][0],np.arange(40)),
                   normalized_gram(features[names[0]][0][:40]),atol=1e-10)
for repeat in range(10):
    seed=SEED+1000+repeat
    permutation=np.random.default_rng(seed).permutation(1000)
    for a,b in PAIRS:
        shuffled=grams[b][:,permutation,:][:,:,permutation].reshape(7,-1)
        m=flat[a]@shuffled.T
        shuffle_matrices[(a,b)].append(m)
        for i in range(7):
            for j in range(7): check_rows.append({'kind':'mismatched_rows','model_a':a,'model_b':b,
                'slot_a':i+1,'slot_b':j+1,'n':1000,'repeat':repeat+1,'seed':seed,'cka':float(m[i,j])})
    print('错配对照完成',repeat+1,'/10',flush=True)
    del shuffled
check_df=pd.DataFrame(check_rows)
check_df.to_csv(OUT/'results'/'cka_checks.csv',index=False)
diagnostic=[]
for (a,b),m in matrices.items():
    sh=np.stack(shuffle_matrices[(a,b)])
    for n in [250,500,1000]:
        diagnostic.append({'pair':f'{a} / {b}','n':n,
            'same_slot_mean':float(np.diag(stability[(a,b,n)]).mean()),
            'max_absolute_change_from_1000':float(np.abs(stability[(a,b,n)]-m).max())})
display(pd.DataFrame(diagnostic))
fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
for ax,((a,b),m) in zip(axes,matrices.items()):
    sh=np.stack(shuffle_matrices[(a,b)])
    lo=np.diag(sh.min(axis=0)); hi=np.diag(sh.max(axis=0))
    ax.fill_between(range(1,8),lo,hi,alpha=.3,label='10 mismatches: min-max')
    ax.plot(range(1,8),np.diag(m),'o-',label='Correct pairing')
    ax.set(title=f'{a} / {b}',xlabel='Same sampling slot',ylabel='CKA',ylim=(0,1))
    ax.legend(fontsize=8)
fig.savefig(OUT/'figures'/'cka_mismatch_controls.png',dpi=180)
display(fig); plt.close(fig)
''')

md(r'''
## 8. 根据真实结果生成的解释

下面的文字由上面实际数值生成，讨论浅／中／深的**采样组**与跨槽位最大值，不宣称它们具有相同感受野或语义。
最大值是在49个组合中事后找到的描述性结果，不是预先指定的统计检验。
''')
code(r'''
lines=['# ImageNet Linear CKA 实际结果与解释',
       f"数据：{REPO}，版本 `{data_manifest['revision']}`，validation，1000类各1张。",
       '选图：固定流式顺序中每类第一张可解码、字节唯一的图片；非均匀随机样本。',
       '模型：三个IMAGENET1K_V1；表示：七个原始卷积输出的全部通道GAP；方法：普通中心化Linear CKA。']
for (a,b),m in matrices.items():
    i,j=np.unravel_index(np.argmax(m),m.shape)
    diagonal=np.diag(m)
    groups=[float(diagonal[:2].mean()),float(diagonal[2:5].mean()),float(diagonal[5:].mean())]
    sh=np.stack(shuffle_matrices[(a,b)])
    gaps=diagonal-np.diag(sh.mean(axis=0))
    count=int(np.sum(diagonal>np.diag(sh.max(axis=0))))
    delta250=float(np.abs(stability[(a,b,250)]-m).max())
    delta500=float(np.abs(stability[(a,b,500)]-m).max())
    trend='较高' if groups[0]>groups[2] else '较低' if groups[0]<groups[2] else '相同'
    best_columns=m.argmax(axis=1)
    gains=m[np.arange(7),best_columns]-diagonal
    example=int(np.argmax(gains)); example_col=int(best_columns[example])
    increases=int(np.sum(np.diff(diagonal)>0))
    lines.extend([f'\n## {a} 与 {b}',
        f'最高组合：{SPECS[a][2][i]}（L{i+1:02d}）与 {SPECS[b][2][j]}（L{j+1:02d}），CKA={m[i,j]:.4f}。',
        f'同槽位浅层L01–L02、中层L03–L05、深层L06–L07的均值依次为 {groups[0]:.4f}、{groups[1]:.4f}、{groups[2]:.4f}；浅层组的均值{trend}。',
        f'最后槽位L07对L07为 {m[6,6]:.4f}。同槽位曲线有{increases}处向上变化，不能把组均值趋势说成每层都下降。',
        f'跨槽位例子：{a} L{example+1:02d} 与 {b} L{example_col+1:02d} 的CKA为 {m[example,example_col]:.4f}，而与 {b} 同槽位L{example+1:02d}的分数是 {diagonal[example]:.4f}。这说明直接按槽位比较会漏掉部分关系，并非证明语义层已严格对齐。',
        f'相对于1000张，250／500张矩阵的最大绝对变化为 {delta250:.4f}／{delta500:.4f}。规模与类别覆盖同时变化。',
        f'7个同槽位中有{count}个正确配对值高于10次错配对照的最大值；相对错配均值的差范围为 {gaps.min():.4f} 至 {gaps.max():.4f}。该对照不等于显著性证明。'])
lines.extend(['\n## 初学者怎样理解这些结果？',
 'CKA=0.9957这样的值表示特定表示的关系结构很接近，不是99.57%分类正确，也不是99.57%的神经元相同。',
 '三对模型均在L01对L01达到本矩阵最高值，且浅层采样组均值高于深层组；同时中间层有上升和跨槽位高值，因此结论应是所选采样组呈现总体差异，而非一条普遍单调规律。',
 '第一层原始卷积是近似线性变换，经过GAP后空间排列被平均，表示可能主要反映全局颜色／亮度统计；这是对高L01分数的一种解释假设，本实验没有另做颜色消融来验证，不能把高分解释为所有边缘检测器相同。',
 'ResNet50与MobileNetV2在L07的相似程度高于涉及VGG16的两对L07结果；这里只能讨论该原始卷积GAP定义下的相似度，不能据此说MobileNetV2更准确或更优。',
 f'9个真实浅／中／深层组合通过另一种等价的特征空间公式复算，最大误差为 {numeric_audit["absolute_error"].max():.2e}。这验证保存特征上的CKA算术，不是独立重放全部CNN前向。'])
lines.extend(['\n## 适用范围与限制',
 '1. 只描述固定ImageNet验证子集的GAP表示关系；不代表完整空间特征等价。',
 '2. 每类仅1张，不能估计类内变化。顺序配额选样和完全重复字节过滤不等于随机抽样或近重复清理。',
 '3. 槽位基于卷积顺序；跨架构的操作角色、通道数和感受野不同。ResNet卷积点在BN及残差相加之前。',
 '4. 普通CKA可能受维度、样本数和主导变化影响。错配也高或规模变化大时，应扩大数据并补充独立抽样／去偏分析，而不是宣布结论稳健。',
 '5. 高相似度不是准确率、因果依据、公平性或模型质量；低相似度不等于没有有效特征。',
 '6. 不能仅用本实验断言浅层普遍相似或深层普遍不同，也不能用CKA给模型排名谁最好。',
 '\n## 团队衔接',
 '本模块提供表示相似度的定量证据，与搭档的通道合成偏好、猫狗个案归因及线性探针互补。',
 '建议从真实数值中选一个最易解释的发现作为meme／视频素材，不把未验证的趋势写进文案。'])
summary='\n\n'.join(lines)
display(Markdown(summary))
(OUT/'summary.md').write_text(summary,encoding='utf-8')
''')

md(r'''
## 9. 实验记录、交付与术语回顾

记录已知尝试：最初系统默认Python 3.7没有torch；定位到5070ti后验证CUDA可用。最初考虑CIFAR10替代，团队明确后撤销；本notebook只使用ImageNet。
官方权重下载曾遇到网络限制／中断，应以实际文件哈希和成功前向为准，部分文件不算完成。更换环境或失败运行时应保留简短记录。

术语回顾：**缓存**是保存以便复用的数值；**manifest**是来源与配置清单；**seed**固定随机过程起点；
**checkpoint**是训练存档，通常不是图片；**logit**是类别原始分数；**probe**是额外训练的分类读出头；
**Grad-CAM**解释预测的空间响应；**activation maximization**优化人工输入激发通道；**CKA**比较样本关系。
只用自己的CKA数值，不把合成图、Grad-CAM颜色或softmax概率当作卷积特征。

交付核心就是本notebook（包含代码、图、表与解释）。输出CSV、图和配置是便于团队复用的附属文件。
团队的两个meme、演示和最终提交另行分工；本模块不替团队虚构这部分交付。

参考：[CKA论文](https://proceedings.mlr.press/v97/kornblith19a.html)、
[作者实现](https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb)、
[ImageNet数据卡及访问条件](https://huggingface.co/datasets/ILSVRC/imagenet-1k)、
[VGG16](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vgg16.html)、
[ResNet50](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html)、
[MobileNetV2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v2.html)。
本地依据：`背景资料/Assignment2-2026A.ipynb`、Lecture5b、Lecture6a、`搭档已完成/`的模型及两图实验。
''')
code(r'''
config={'complete':True,'environment':environment,'seed':SEED,'batch_size':BATCH_SIZE,
    'dataset_manifest_sha256':sha256_file(data_manifest_path),'dataset':{
        k:v for k,v in data_manifest.items() if k!='images'},
    'image_ids':IMAGE_IDS,'models':model_resources,'preprocess':preprocess_config,
    'representation':'all-channel GAP of raw Conv2d output before subsequent BN/ReLU; no probe standardization',
    'estimator':'ordinary centered linear CKA','main_scores':len(results),
    'formula_checks':{k:bool(v) for k,v in checks.items()},'extraction':extraction_stats,
    'real_feature_numeric_audit':{'combinations':len(numeric_audit),'max_absolute_error':float(numeric_audit['absolute_error'].max())},
    'controls':{'sample_sizes':[250,500,1000],'mismatch_repetitions':10},
        'result_sha256':sha256_file(OUT/'results'/'cka_main.csv'),
    'controls_sha256':sha256_file(OUT/'results'/'cka_checks.csv')}
(OUT/'experiment_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print('完成：21个层表示，147个主分数；公式检查通过；规模与错配对照完成。')
print('附属结果目录：',OUT)
print('重新运行优先校验缓存；所有结论均由真实结果生成。')
''')

notebook={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
    'kernelspec':{'display_name':'Python (5070ti)','language':'python','name':'5070ti'},
    'language_info':{'name':'python','version':'3.10.19'},
    'interpreter':{'path':r'C:\Anaconda3\envs\5070ti\python.exe'},
    'task_status':'Prepared; full execution requires authorized ImageNet access and completed model downloads'}}
path=ROOT/'A2-cka-study.ipynb'
path.write_text(json.dumps(notebook,ensure_ascii=False,indent=1),encoding='utf-8')
print('Built:',path,'cells:',len(cells),'code syntax checks passed')
