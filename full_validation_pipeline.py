"""Checkpointed full ImageNet validation study; also embedded in the notebooks."""
# %% environment
import os,sys,json,time,hashlib,logging,gc,copy,warnings,math,argparse,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ['HF_HUB_DISABLE_PROGRESS_BARS']='1'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING']='1'
ROOT=Path.cwd().resolve()
if not (ROOT/'resource_status.json').exists() and (ROOT.parent/'resource_status.json').exists():ROOT=ROOT.parent
FULL=ROOT/'full_validation_study'
FULL.mkdir(exist_ok=True)
os.environ['HF_HUB_CACHE']=str(ROOT/'.cache/huggingface/hub')
os.environ['HF_XET_CACHE']=str(ROOT/'.cache/huggingface/xet')
os.environ['HF_DATASETS_CACHE']=str(ROOT/'.cache/huggingface/datasets')
os.environ.setdefault('OMP_NUM_THREADS','8')
os.environ.setdefault('MKL_NUM_THREADS','8')
import numpy as np,pandas as pd
import torch,torchvision
from torchvision import models
import torch.nn.functional as F
from PIL import Image
import pyarrow.parquet as pq
from huggingface_hub import HfApi,hf_hub_download,get_token
logging.getLogger('huggingface_hub').setLevel(logging.ERROR)
torch.hub.set_dir(str(ROOT/'.cache/torch/hub'))
torch.set_num_threads(8)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert DEVICE.type=='cuda','Full study requires the verified CUDA environment.'
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True,warn_only=True)
SEED=5495
torch.manual_seed(SEED)
np.random.seed(SEED)
N_FULL=50000
REVISION='49e2ee26f3810fb5a7536bbf732a7b07389a47b5'
REPO='ILSVRC/imagenet-1k'
SPECS={
 'vgg16':(models.vgg16,models.VGG16_Weights.IMAGENET1K_V1,
 ['features.0','features.5','features.10','features.14','features.19','features.24','features.28']),
 'resnet50':(models.resnet50,models.ResNet50_Weights.IMAGENET1K_V1,
 ['conv1','layer1.2.conv2','layer2.2.conv1','layer3.0.conv3','layer3.3.conv2','layer4.0.conv1','layer4.2.conv3']),
 'mobilenet_v2':(models.mobilenet_v2,models.MobileNet_V2_Weights.IMAGENET1K_V1,
 ['features.0.0','features.4.conv.0.0','features.6.conv.2','features.9.conv.2','features.12.conv.1.0','features.15.conv.1.0','features.18.0'])}
TRANSFORM=models.VGG16_Weights.IMAGENET1K_V1.transforms()
COUNTS=np.rint(np.linspace(0,224*224,21)).astype(np.int64)
FRACTIONS=COUNTS/(224*224)
METHODS=[f'L{i:02d}' for i in range(1,8)]+['random']
CONFIG={'engine_version':'full_validation_v2','dataset':REPO,'revision':REVISION,'split':'validation',
 'n_cka':50000,'n_insertion_deletion':50000,'layers':7,'seed':SEED,'counts':COUNTS.tolist(),
 'preprocess':str(TRANSFORM),'target':'FP32 original Top-1, fixed across all perturbations',
 'baseline':'ImageNet mean RGB (zero normalized tensor)','auc':'probability trapezoidal area',
 'gradcam_precision':'FP32','perturbation_precision':'FP16 only after measured pilot gate; otherwise FP32',
 'cam_backward':'FP32 CUDA autograd; deterministic algorithms off in the performance stage; no bitwise guarantee',
 'id_cuda_performance':'cuDNN benchmark on; deterministic algorithms off; fixed seeds, no bitwise guarantee',
 'cka':'exact centered feature-space covariance, float64; no NxN matrices',
 'random_ties':'seeded per-image random initial order; stable descending CAM sort',
 'bootstrap':'1000 paired class-cluster resamples; exploratory class-composition sensitivity'}

def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}.{threading.get_ident()}')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    with temporary.open('r+b') as stream:
        stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()

def progress(stage,done,total,message='',**extra):
    record={'stage':stage,'done':int(done),'total':int(total),'message':message,
            'updated_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),**extra}
    write_json(FULL/'progress.json',record)
    print(json.dumps(record,ensure_ascii=False),flush=True)

# %% dataset
def prepare_full_dataset():
    manifest_path=FULL/'data/manifest.json'
    if manifest_path.exists():
        m=json.loads(manifest_path.read_text(encoding='utf-8'))
        assert len(m['images'])==N_FULL and m['revision']==REVISION
        assert np.bincount([r['label'] for r in m['images']],minlength=1000).tolist()==[50]*1000
        if all((ROOT/r['path']).exists() for r in m['images']):return m
        print('Manifest retained as evidence; image cache absent. Rebuilding authorized data cache.',flush=True)
    if not get_token():raise RuntimeError('Local authorized Hugging Face login is required.')
    parquet_dir=FULL/'data/parquet';parquet_dir.mkdir(parents=True,exist_ok=True)
    info=HfApi().dataset_info(REPO,revision=REVISION,files_metadata=True,token=True)
    shards=sorted([s for s in info.siblings if 'validation-' in s.rfilename and s.rfilename.endswith('.parquet')],key=lambda s:s.rfilename)
    assert len(shards)==14
    expected=[]
    for s in shards:
        lfs=s.lfs
        sha=lfs.get('sha256') if isinstance(lfs,dict) else getattr(lfs,'sha256',None)
        expected.append({'file':s.rfilename,'size':s.size,'sha256':sha})
    write_json(FULL/'data/download_manifest.json',{'repo':REPO,'revision':REVISION,'shards':expected,
          'bytes_total':sum(x['size'] for x in expected)})
    progress('download',0,len(shards),f"Official validation parquet: {sum(x['size'] for x in expected)/1024**3:.2f} GiB")
    def fetch(item):
        final=parquet_dir/Path(item['file']).name
        if not final.exists() or final.stat().st_size!=item['size']:
            import requests
            url=f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{item["file"]}'
            partial=final.with_name(final.name+'.partial')
            for attempt in range(8):
                try:
                    if partial.exists() and partial.stat().st_size==item['size']:
                        if not item['sha256'] or digest(partial)==item['sha256']:
                            partial.replace(final);break
                        partial.rename(partial.with_name(partial.name+f'.corrupt.{int(time.time())}'))
                    offset=partial.stat().st_size if partial.exists() else 0
                    headers={'Authorization':'Bearer '+get_token()}
                    if offset:headers['Range']=f'bytes={offset}-'
                    # requests removes Authorization when redirecting to the CDN host.
                    with requests.get(url,headers=headers,stream=True,timeout=(20,30)) as response:
                        response.raise_for_status()
                        resume=response.status_code==206 and offset>0
                        if resume:assert response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
                        count=offset if resume else 0;next_log=count+64*1024**2
                        with partial.open('ab' if resume else 'wb') as output:
                            for block in response.iter_content(1024**2):
                                if not block:continue
                                output.write(block);count+=len(block)
                                if count>=next_log:
                                    print('Download',final.name,round(count/1024**2),'MiB',flush=True)
                                    next_log=count+64*1024**2
                    assert partial.stat().st_size==item['size']
                    actual=digest(partial)
                    if item['sha256']:assert actual==item['sha256']
                    partial.replace(final);break
                except Exception as exc:
                    print('Retry official shard',final.name,'attempt',attempt+1,'error',type(exc).__name__,flush=True)
                    if attempt==7:raise RuntimeError('Official validation shard download failed: '+final.name) from None
                    time.sleep(min(2**attempt,16))
        assert final.stat().st_size==item['size']
        actual=digest(final)
        if item['sha256']:assert actual==item['sha256'],final.name
        return final,actual
    completed=0;verified={}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(fetch,x):x for x in expected}
        for future in as_completed(futures):
            path,sha=future.result();verified[path.name]=sha;completed+=1
            progress('download',completed,14,path.name)
    images_dir=FULL/'data/images';images_dir.mkdir(exist_ok=True)
    records=[];bad=[];seen={};index=0;started=time.monotonic()
    for item in expected:
        path=parquet_dir/Path(item['file']).name
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128,columns=['image','label']):
            for row in batch.to_pylist():
                raw=row['image']['bytes'];label=int(row['label'])
                sha=hashlib.sha256(raw).hexdigest()
                image_path=images_dir/f'{index:05d}.img'
                if not image_path.exists() or image_path.stat().st_size!=len(raw):image_path.write_bytes(raw)
                try:
                    with Image.open(image_path) as im:
                        im.convert('RGB').load();width,height=im.size
                except (OSError,ValueError) as exc:
                    bad.append({'row':index,'error':str(exc)})
                    raise RuntimeError('A validation image cannot be decoded; no silent exclusion.') from exc
                records.append({'image_id':f'imagenet-validation-{index:05d}','row_index':index,
                    'label':label,'sha256':sha,'path':str(image_path.relative_to(ROOT)),
                    'width':width,'height':height})
                seen[sha]=seen.get(sha,0)+1;index+=1
                if index%2500==0:progress('decode',index,N_FULL,'Validated original JPEG bytes',seconds=time.monotonic()-started)
    assert index==N_FULL
    assert np.bincount([r['label'] for r in records],minlength=1000).tolist()==[50]*1000
    old_path=ROOT/'cka_study/imagenet_subset/manifest.json'
    if old_path.exists():
        old=json.loads(old_path.read_text(encoding='utf-8'))
        for r in old['images']:assert records[r['row_index']]['sha256']==r['sha256']
    # Preserve all official rows, including any source duplicates; report, do not remove.
    np.random.default_rng(SEED).shuffle(records)
    m={'repo':REPO,'revision':REVISION,'split':'validation','selection':'entire official validation split; seeded final order',
       'n':N_FULL,'classes':1000,'per_class':50,'unique_byte_hashes':len(seen),
       'duplicate_byte_rows':N_FULL-len(seen),'invalid_images':bad,'parquet_sha256':verified,'images':records}
    write_json(manifest_path,m)
    progress('data_ready',N_FULL,N_FULL,'Complete official validation set verified')
    return m

# %% extraction
class FullImages(torch.utils.data.Dataset):
    def __init__(self,manifest):self.records=manifest['images']
    def __len__(self):return len(self.records)
    def __getitem__(self,index):
        r=self.records[index]
        with Image.open(ROOT/r['path']) as im:x=TRANSFORM(im.convert('RGB'))
        return x,index

def model_resources():
    result={}
    for name,(_,weights,layers) in SPECS.items():
        p=Path(torch.hub.get_dir())/'checkpoints'/weights.url.rsplit('/',1)[1]
        if not p.exists():
            torch.hub.load_state_dict_from_url(weights.url,model_dir=str(p.parent),check_hash=True,progress=False,map_location='cpu')
        sha=digest(p);assert sha.startswith(p.stem.rsplit('-',1)[1])
        result[name]={'weights':str(weights),'sha256':sha,'layers':layers,'url':weights.url}
    return result

def extract_features(manifest,resources):
    dataset=FullImages(manifest);features={};metadata={}
    for name,(builder,weights,layers) in SPECS.items():
        folder=FULL/'features'/name;folder.mkdir(parents=True,exist_ok=True)
        meta_path=folder/'metadata.json'
        signature={'engine':CONFIG['engine_version'],'data':digest(FULL/'data/manifest.json'),
                   'model':resources[name],'precision':'FP32, channels_last, TF32 disabled',
                   'representation':'all-channel raw Conv2d GAP before subsequent BN/ReLU',
                   'torch':torch.__version__,'torchvision':torchvision.__version__}
        if meta_path.exists():
            old=json.loads(meta_path.read_text(encoding='utf-8'))
            assert old['signature']==signature
            present=all((folder/f'L{i:02d}.npy').exists() for i in range(1,8))
            if old.get('complete') and present:
                features[name]=[np.load(folder/f'L{i:02d}.npy',mmap_mode='r') for i in range(1,8)]
                metadata[name]=old;continue
            offset=old['done'] if present and not old.get('complete') else 0
        else:offset=0
        model=builder(weights=weights,progress=False).eval().to(DEVICE,memory_format=torch.channels_last)
        for p in model.parameters():p.requires_grad_(False)
        modules=dict(model.named_modules());channels=[modules[layer].out_channels for layer in layers]
        arrays=[np.lib.format.open_memmap(folder/f'L{i+1:02d}.npy',mode='r+' if offset else 'w+',
                    dtype=np.float32,shape=(N_FULL,c)) for i,c in enumerate(channels)]
        current={};handles=[];shapes={}
        def capture(slot):
            def hook(module,inputs,output):
                shapes[slot]=list(output.shape[1:])
                current[slot]=output.detach().clone().mean((2,3)).float().cpu().numpy()
            return hook
        for i,layer in enumerate(layers):handles.append(modules[layer].register_forward_hook(capture(i)))
        loader=torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,range(offset,N_FULL)),
            batch_size=32,shuffle=False,num_workers=0,pin_memory=True)
        started=time.monotonic();done=offset
        try:
            with torch.no_grad():
                for x,idx in loader:
                    current.clear();out=model(x.to(DEVICE,non_blocking=True,memory_format=torch.channels_last))
                    assert torch.isfinite(out).all().item()
                    indices=idx.numpy()
                    assert np.array_equal(indices,np.arange(done,done+len(indices)))
                    for i,a in enumerate(arrays):
                        assert np.isfinite(current[i]).all();a[indices]=current[i]
                    done+=len(indices)
                    if done%512==0 or done==N_FULL:
                        for a in arrays:a.flush()
                        write_json(meta_path,{'signature':signature,'complete':False,'done':done,
                            'channels':channels,'raw_shapes':shapes})
                        progress('features_'+name,done,N_FULL,elapsed_s=time.monotonic()-started)
            final={'signature':signature,'complete':True,'done':done,'channels':channels,'raw_shapes':shapes,
                   'sha256':{f'L{i+1:02d}':digest(folder/f'L{i+1:02d}.npy') for i in range(7)}}
            write_json(meta_path,final);metadata[name]=final
        finally:
            for h in handles:h.remove()
            del model,current,arrays;gc.collect();torch.cuda.empty_cache()
        features[name]=[np.load(folder/f'L{i:02d}.npy',mmap_mode='r') for i in range(1,8)]
    return features,metadata

# %% exact CKA
def covariance_cka(features,indices=None,permutations=None):
    boundaries={};tensors={};norms={}
    with torch.no_grad():
        for name,values in features.items():
            selected=[np.asarray(x if indices is None else x[indices]) for x in values]
            joined=np.concatenate(selected,axis=1)
            x=torch.from_numpy(joined).to(DEVICE,dtype=torch.float64)
            x-=x.mean(0,keepdim=True)
            boundaries[name]=np.cumsum([0]+[v.shape[1] for v in values])
            tensors[name]=x
            b=boundaries[name];norm=[]
            for i in range(7):
                z=x[:,b[i]:b[i+1]];cov=z.T@z;norm.append(torch.linalg.matrix_norm(cov))
            norms[name]=torch.stack(norm)
            del joined,selected,cov
        names=list(features);pairs=[(names[0],names[1]),(names[0],names[2]),(names[1],names[2])]
        def calculate(permutation=None):
            result={}
            for a,b in pairs:
                y=tensors[b] if permutation is None else tensors[b].index_select(0,permutation)
                cov=tensors[a].T@y
                scores=torch.empty((7,7),device=DEVICE,dtype=torch.float64)
                ba,bb=boundaries[a],boundaries[b]
                for i in range(7):
                    for j in range(7):
                        block=cov[ba[i]:ba[i+1],bb[j]:bb[j+1]]
                        scores[i,j]=block.square().sum()/(norms[a][i]*norms[b][j])
                result[(a,b)]=scores.cpu().numpy()
                del cov,y
            return result
        primary=calculate()
        controls=[]
        for repeat in range(permutations or 0):
            perm=np.random.default_rng(SEED+1000+repeat).permutation(len(next(iter(tensors.values()))))
            controls.append(calculate(torch.from_numpy(perm).to(DEVICE)))
            progress('cka_mismatch',repeat+1,permutations,'Exact feature-space calculation')
        del tensors,norms;gc.collect();torch.cuda.empty_cache()
    return primary,controls

def run_cka(manifest,features):
    folder=FULL/'results';folder.mkdir(exist_ok=True)
    path=folder/'cka_results.npz'
    if path.exists() and (folder/'cka_complete.json').exists():return
    # Small-N independent Gram check, never allocate a 50,000 x 50,000 matrix.
    probe=np.arange(200);check,_=covariance_cka(features,probe)
    audit=[]
    for (a,b),m in check.items():
        for i in [0,3,6]:
            x=np.asarray(features[a][i][probe],np.float64);y=np.asarray(features[b][i][probe],np.float64)
            x-=x.mean(0);y-=y.mean(0);k=x@x.T;l=y@y.T
            reference=float((k*l).sum()/np.linalg.norm(k)/np.linalg.norm(l))
            error=abs(reference-m[i,i]);assert error<1e-9
            audit.append({'a':a,'b':b,'slot':i+1,'error':error})
    sizes=[1000,5000,10000,25000,50000];labels=np.array([r['label'] for r in manifest['images']])
    stability={};primary=None;controls=None
    for n in sizes:
        per_class=n//1000
        indices=np.concatenate([np.flatnonzero(labels==c)[:per_class] for c in range(1000)])
        assert len(indices)==n
        matrices,mismatches=covariance_cka(features,indices,permutations=10 if n==N_FULL else None)
        for pair,matrix in matrices.items():stability[(*pair,n)]=matrix
        if n==N_FULL:primary=matrices;controls=mismatches
        progress('cka_size',n,N_FULL,'Balanced per-class subsample')
    rows=[];control_rows=[];values={}
    for (a,b),m in primary.items():
        values[f'{a}__{b}']=m
        for i in range(7):
            for j in range(7):
                rows.append({'model_a':a,'slot_a':i+1,'module_a':SPECS[a][2][i],
                    'model_b':b,'slot_b':j+1,'module_b':SPECS[b][2][j],'cka':float(m[i,j]),'n':N_FULL})
        for n in sizes:
            values[f'{a}__{b}__n{n}']=stability[(a,b,n)]
        for r,control in enumerate(controls):values[f'{a}__{b}__shuffle{r}']=control[(a,b)]
    for (a,b,n),m in stability.items():
        for i in range(7):
            for j in range(7):control_rows.append({'kind':'balanced_size','a':a,'b':b,'n':n,'i':i+1,'j':j+1,'repeat':0,'cka':float(m[i,j])})
    for r,control in enumerate(controls):
        for (a,b),m in control.items():
            for i in range(7):
                for j in range(7):control_rows.append({'kind':'mismatch','a':a,'b':b,'n':N_FULL,'i':i+1,'j':j+1,'repeat':r+1,'cka':float(m[i,j])})
    pd.DataFrame(rows).to_csv(folder/'cka_main.csv',index=False)
    pd.DataFrame(control_rows).to_csv(folder/'cka_controls.csv',index=False)
    pd.DataFrame(audit).to_csv(folder/'cka_numeric_audit.csv',index=False)
    np.savez(path,**values)
    write_json(folder/'cka_complete.json',{'complete':True,'n':N_FULL,'scores':147,'sizes':sizes,
          'mismatches':10,'numeric_audit_max_error':max(r['error'] for r in audit),
          'method':CONFIG['cka'],'results_sha256':digest(path)})

# %% vectorized perturbations
def random_orders(row_indices):
    values=np.stack([np.random.default_rng(SEED+20000+int(i)).permutation(224*224) for i in row_indices])
    return torch.from_numpy(values).to(DEVICE)

def rank_maps(raw_cams,row_indices):
    up=torch.cat([F.interpolate(c[:,None],size=(224,224),mode='bilinear',align_corners=False)
                  for c in raw_cams],dim=1).reshape(len(row_indices),7,-1)
    random_order=random_orders(row_indices)
    initial=random_order[:,None,:].expand(-1,7,-1)
    sorted_positions=torch.argsort(up.gather(2,initial),dim=2,descending=True,stable=True)
    ordered=torch.cat([initial.gather(2,sorted_positions),random_order[:,None,:]],dim=1)
    ranks=torch.empty(ordered.shape,device=DEVICE,dtype=torch.int32)
    ranks.scatter_(2,ordered,torch.arange(224*224,device=DEVICE,dtype=torch.int32)[None,None,:].expand_as(ranks))
    zero=up.max(2).values==0
    return ranks,zero

def evaluate_curves(model,x,ranks,targets,batch_size=128):
    dtype=next(model.parameters()).dtype
    x=x.to(dtype=dtype,memory_format=torch.channels_last)
    b=len(x);curves=torch.empty((b,8,2,21),device=DEVICE,dtype=torch.float32)
    counts=torch.as_tensor(COUNTS,device=DEVICE)
    with torch.inference_mode():
        full=model(x).float().softmax(1).gather(1,targets[:,None])[:,0]
        blank=model(torch.zeros((1,3,224,224),device=DEVICE,dtype=dtype).contiguous(memory_format=torch.channels_last)).float().softmax(1)[0]
        baseline=blank[targets]
        curves[:,:,0,0]=baseline[:,None];curves[:,:,0,-1]=full[:,None]
        curves[:,:,1,0]=full[:,None];curves[:,:,1,-1]=baseline[:,None]
        total=b*8*2*19
        for start in range(0,total,batch_size):
            state=torch.arange(start,min(start+batch_size,total),device=DEVICE)
            image=state//(8*2*19);method=(state//(2*19))%8;metric=(state//19)%2;step=state%19+1
            mask=ranks[image,method,:]<counts[step,None]
            mask=torch.logical_xor(mask,metric[:,None].bool()).reshape(-1,1,224,224)
            perturbed=(x[image]*mask).contiguous(memory_format=torch.channels_last)
            probability=model(perturbed).float().softmax(1).gather(1,targets[image,None])[:,0]
            curves[image,method,metric,step]=probability
    assert torch.isfinite(curves).all().item()
    return curves

def collect_cams(model,x,layers):
    captured={};snapshots={};handles=[];modules=dict(model.named_modules())
    def capture(slot):
        def hook(module,inputs,output):captured[slot]=output;snapshots[slot]=output.detach().clone()
        return hook
    for i,layer in enumerate(layers):handles.append(modules[layer].register_forward_hook(capture(i)))
    try:
        x=x.detach().requires_grad_(True)
        logits=model(x);targets=logits.argmax(1)
        gradients=torch.autograd.grad(logits.gather(1,targets[:,None]).sum(),[captured[i] for i in range(7)])
        cams=[F.relu((grad.mean((2,3),keepdim=True)*snapshots[i]).sum(1)).detach() for i,grad in enumerate(gradients)]
        probabilities=logits.detach().softmax(1).gather(1,targets[:,None])[:,0]
        return cams,targets.detach(),probabilities
    finally:
        for handle in handles:handle.remove()

def validate_precision(name,model_fp32,model_fp16):
    folder=FULL/'precision';folder.mkdir(exist_ok=True)
    path=folder/f'{name}.json'
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if saved.get('engine_version')==CONFIG['engine_version']:return saved
    old_manifest=json.loads((ROOT/'cka_study/imagenet_subset/manifest.json').read_text(encoding='utf-8'))
    with np.load(ROOT/'cka_study/insertion_deletion'/f'{name}_id.npz',allow_pickle=False) as saved:
        maps=saved['cams'][:24].copy();targets=saved['targets'][:24].copy()
    x=[]
    for r in old_manifest['images'][:24]:
        with Image.open(ROOT/r['path']) as im:x.append(TRANSFORM(im.convert('RGB')))
    x=torch.stack(x).to(DEVICE,memory_format=torch.channels_last)
    source=[torch.from_numpy(maps[:,i]).to(DEVICE) for i in range(7)]
    ranks,_=rank_maps(source,[r['row_index'] for r in old_manifest['images'][:24]])
    targets=torch.from_numpy(targets).to(DEVICE)
    started=time.monotonic();reference=evaluate_curves(model_fp32,x,ranks,targets)
    ref_seconds=time.monotonic()-started
    started=time.monotonic();approx=evaluate_curves(model_fp16,x,ranks,targets)
    torch.cuda.synchronize();half_seconds=time.monotonic()-started
    error=(reference-approx).abs().cpu().numpy()
    r=reference.cpu().numpy();a=approx.cpu().numpy()
    auc_error=np.abs(np.trapezoid(r.astype(np.float64),x=FRACTIONS,axis=-1)-np.trapezoid(a.astype(np.float64),x=FRACTIONS,axis=-1))
    passed=bool(error.max()<.05 and np.quantile(error,.995)<.015 and auc_error.mean()<.003 and auc_error.max()<.015)
    result={'engine_version':CONFIG['engine_version'],'model':name,'images':24,'max_curve_error':float(error.max()),'p995_curve_error':float(np.quantile(error,.995)),
      'mean_auc_error':float(auc_error.mean()),'max_auc_error':float(auc_error.max()),'fp16_passed':passed,
      'fp32_seconds':ref_seconds,'fp16_seconds':half_seconds,
      'gate':'max curve < .05; 99.5% curve < .015; mean AUC < .003; max AUC < .015',
      'meaning':'empirical precision check, not a worst-case error guarantee over all validation images'}
    write_json(path,result);progress('precision_'+name,24,24,json.dumps(result))
    return result

# %% full I/D
def run_insertion_deletion(manifest,resources):
    torch.backends.cudnn.benchmark=True
    torch.backends.cudnn.deterministic=False
    torch.use_deterministic_algorithms(False)
    dataset=FullImages(manifest);n=len(dataset);chunk_size=64
    precision={}
    for name,(builder,weights,layers) in SPECS.items():
        folder=FULL/'insertion_deletion'/name;folder.mkdir(parents=True,exist_ok=True)
        signature={'config':CONFIG,'data':digest(FULL/'data/manifest.json'),'model':resources[name]}
        signature_path=folder/'signature.json'
        if signature_path.exists():assert json.loads(signature_path.read_text(encoding='utf-8'))==signature
        else:write_json(signature_path,signature)
        if (folder/'complete.json').exists() and all(
                (folder/f'part_{start:05d}_{min(start+chunk_size,n):05d}.npz').exists()
                for start in range(0,n,chunk_size)):
            precision[name]=json.loads((FULL/'precision'/f'{name}.json').read_text(encoding='utf-8'));continue
        fp32=builder(weights=weights,progress=False).eval().to(DEVICE,memory_format=torch.channels_last)
        for p in fp32.parameters():p.requires_grad_(False)
        first=dataset[0][0][None].to(DEVICE,memory_format=torch.channels_last)
        with torch.no_grad():before=fp32(first)
        for module in fp32.modules():
            if isinstance(module,(torch.nn.ReLU,torch.nn.ReLU6)):module.inplace=False
        with torch.no_grad():after=fp32(first)
        equiv=float((before-after).abs().max().item());assert torch.allclose(before,after,atol=1e-6,rtol=1e-6)
        fp16=copy.deepcopy(fp32).half()
        precision[name]=validate_precision(name,fp32,fp16)
        evaluator=fp16 if precision[name]['fp16_passed'] else fp32
        write_json(folder/'execution_protocol.json',{'precision':precision[name],'forward_equivalence_error':equiv,
                  'batch_size_perturbation':128,'cam_batch_size':8,'chunk_size':chunk_size})
        started=time.monotonic();completed_this_run=0;non_determinism=set()
        for start in range(0,n,chunk_size):
            end=min(start+chunk_size,n);part=folder/f'part_{start:05d}_{end:05d}.npz';meta=part.with_suffix('.json')
            if part.exists() and meta.exists():
                try:
                    saved=json.loads(meta.read_text(encoding='utf-8'))
                    if (saved['start'],saved['end'],saved['n'])!=(start,end,end-start):
                        raise ValueError('Checkpoint range mismatch')
                    if saved['sha256']!=digest(part):raise ValueError('Checkpoint checksum mismatch')
                except (ValueError,KeyError,TypeError) as error:
                    # Preserve interrupted/corrupt artifacts, then recompute this shard only.
                    suffix=f'.invalid.{time.time_ns()}'
                    for artifact in (meta,part):artifact.replace(artifact.with_name(artifact.name+suffix))
                    print(f'Recovering {name} shard {start}:{end}: {error}',flush=True)
                else:
                    continue
            curves=[];targets_all=[];references=[];raw_parts=[[] for _ in range(7)];zero_parts=[];fallback=0
            for batch_start in range(start,end,8):
                idx=list(range(batch_start,min(batch_start+8,end)))
                x=torch.stack([dataset[i][0] for i in idx]).to(DEVICE,memory_format=torch.channels_last)
                with warnings.catch_warnings(record=True) as logged:
                    warnings.simplefilter('always')
                    cams,targets,ref=collect_cams(fp32,x,layers)
                for warning in logged:non_determinism.add(str(warning.message).split(' (')[0][:300])
                ranks,zero=rank_maps(cams,[manifest['images'][i]['row_index'] for i in idx])
                for slot,cam in enumerate(cams):raw_parts[slot].append(cam.cpu().numpy())
                del cams
                scores=evaluate_curves(evaluator,x,ranks,targets)
                bad=(scores[:,0,0,-1]-ref).abs()>.01
                if evaluator is fp16 and bad.any().item():
                    # Original-score precision outliers are evaluated in FP32 instead.
                    scores[bad]=evaluate_curves(fp32,x[bad],ranks[bad],targets[bad]);fallback+=int(bad.sum().item())
                assert torch.isfinite(scores).all().item() and scores.min().item()>=0 and scores.max().item()<=1+1e-6
                for local,slot in torch.nonzero(zero,as_tuple=False).cpu().tolist():
                    assert torch.equal(ranks[local,slot],ranks[local,7])
                    scores[local,slot]=scores[local,7]
                curves.append(scores.cpu().numpy());targets_all.append(targets.cpu().numpy())
                references.append(ref.cpu().numpy());zero_parts.append(zero.cpu().numpy())
                del x,scores,ranks,targets,ref,zero
            fields={'curves':np.concatenate(curves),'targets':np.concatenate(targets_all),
                'fp32_original_prob':np.concatenate(references),'all_zero':np.concatenate(zero_parts),
                'indices':np.arange(start,end),'labels':np.array([manifest['images'][i]['label'] for i in range(start,end)])}
            for slot,parts in enumerate(raw_parts):fields[f'raw_cam_{slot+1:02d}']=np.concatenate(parts)
            temporary=part.with_name(part.name+'.tmp')
            with temporary.open('wb') as stream:
                np.savez(stream,**fields);stream.flush();os.fsync(stream.fileno())
            temporary.replace(part)
            write_json(meta,{'sha256':digest(part),'start':start,'end':end,'n':end-start,
                'zero_maps':int(fields['all_zero'].sum()),'fp32_fallback_images':fallback,
                'backward_warnings':sorted(non_determinism),'fixed_target':True})
            completed_this_run+=end-start;elapsed=time.monotonic()-started
            progress('id_'+name,end,n,'Durable shard committed',elapsed_s=elapsed,
                     eta_s=(n-end)*elapsed/max(completed_this_run,1),precision=str(next(evaluator.parameters()).dtype))
            del fields,curves,raw_parts
        write_json(folder/'complete.json',{'complete':True,'n':n,'parts':math.ceil(n/chunk_size),
            'precision':precision[name],'backward_warnings':sorted(non_determinism)})
        del fp32,fp16,evaluator;gc.collect();torch.cuda.empty_cache()
    return precision

# %% aggregation
def aggregate_id(manifest):
    result_dir=FULL/'results';result_dir.mkdir(exist_ok=True)
    bootstrap=np.random.default_rng(SEED+30000).integers(0,1000,size=(1000,1000))
    summary=[];curve_means={};audit=[]
    for name in SPECS:
        folder=FULL/'insertion_deletion'/name
        assert (folder/'complete.json').exists()
        aucs=np.empty((N_FULL,8,2),np.float64);labels=np.empty(N_FULL,np.int64)
        targets=np.empty(N_FULL,np.int64);zero_count=np.zeros(7,np.int64)
        summed=np.zeros((8,2,21),np.float64);count=0;fallbacks=0;max_original_error=0.
        for part in sorted(folder.glob('part_*.npz')):
            metadata=json.loads(part.with_suffix('.json').read_text(encoding='utf-8'))
            assert metadata['sha256']==digest(part)
            with np.load(part,allow_pickle=False) as arrays:
                indices=arrays['indices'];assert np.array_equal(indices,np.arange(count,count+len(indices)))
                curves=arrays['curves'].astype(np.float64)
                assert curves.shape==(len(indices),8,2,21) and np.isfinite(curves).all()
                assert np.allclose(curves[:,:,0,0],curves[:,:,1,-1],atol=1e-7)
                assert np.allclose(curves[:,:,1,0],curves[:,:,0,-1],atol=1e-7)
                # Independent manual trapezoidal summation on every saved curve.
                areas=((curves[:,:,:,1:]+curves[:,:,:,:-1])*.5*np.diff(FRACTIONS)).sum(-1)
                assert np.allclose(areas,np.trapezoid(curves,x=FRACTIONS,axis=-1),atol=1e-12)
                aucs[indices]=areas;labels[indices]=arrays['labels'];targets[indices]=arrays['targets']
                summed+=curves.sum(0);zero_count+=arrays['all_zero'].sum(0)
                max_original_error=max(max_original_error,float(np.abs(curves[:,0,0,-1]-arrays['fp32_original_prob']).max()))
                count+=len(indices);fallbacks+=metadata['fp32_fallback_images']
        assert count==N_FULL and np.bincount(labels,minlength=1000).tolist()==[50]*1000
        class_aucs=np.stack([aucs[labels==c].mean(0) for c in range(1000)])
        for k,method in enumerate(METHODS):
            ins_gain=class_aucs[:,k,0]-class_aucs[:,7,0]
            del_gain=class_aucs[:,7,1]-class_aucs[:,k,1]
            ci_ins=np.quantile(ins_gain[bootstrap].mean(1),[.025,.975])
            ci_del=np.quantile(del_gain[bootstrap].mean(1),[.025,.975])
            summary.append({'model':name,'method':method,'n_images':N_FULL,'n_classes':1000,
                'insertion_auc_mean':float(aucs[:,k,0].mean()),'deletion_auc_mean':float(aucs[:,k,1].mean()),
                'insertion_gain_vs_random':float(ins_gain.mean()),'ins_gain_ci_low':float(ci_ins[0]),'ins_gain_ci_high':float(ci_ins[1]),
                'deletion_gain_vs_random':float(del_gain.mean()),'del_gain_ci_low':float(ci_del[0]),'del_gain_ci_high':float(ci_del[1]),
                'all_zero_maps':int(zero_count[k]) if k<7 else 0})
        curve_means[name]=summed/N_FULL
        np.savez(result_dir/f'{name}_per_image_auc.npz',auc=aucs,labels=labels,targets=targets)
        audit.append({'model':name,'n':count,'zero_maps':int(zero_count.sum()),'fp32_fallback_images':fallbacks,
                      'max_full_input_probability_error':max_original_error,'curve_endpoint_checks':True,'manual_auc_checks':True})
    pd.DataFrame(summary).to_csv(result_dir/'id_summary.csv',index=False)
    pd.DataFrame(audit).to_csv(result_dir/'id_numeric_audit.csv',index=False)
    np.savez(result_dir/'id_mean_curves.npz',**curve_means)
    return summary,audit

def make_figures(manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig_dir=FULL/'figures';fig_dir.mkdir(exist_ok=True)
    names=list(SPECS);pairs=[(names[0],names[1]),(names[0],names[2]),(names[1],names[2])]
    with np.load(FULL/'results/cka_results.npz') as values:
        fig,axes=plt.subplots(1,3,figsize=(18,5),constrained_layout=True)
        for ax,(a,b) in zip(axes,pairs):
            m=values[f'{a}__{b}'];image=ax.imshow(m,vmin=0,vmax=1,cmap='viridis')
            ax.set(title=f'{a} vs {b}: 50,000 images',xlabel=b,ylabel=a)
            ax.set_xticks(range(7),METHODS[:7]);ax.set_yticks(range(7),METHODS[:7])
            for i in range(7):
                for j in range(7):ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if m[i,j]<.5 else 'black')
        fig.colorbar(image,ax=axes,label='Exact Linear CKA (FP64 feature-space covariance)')
        fig.savefig(fig_dir/'cka_full_matrices.png',dpi=170);plt.close(fig)
        fig,axes=plt.subplots(1,3,figsize=(16,4),constrained_layout=True)
        for ax,(a,b) in zip(axes,pairs):
            controls=np.stack([np.diag(values[f'{a}__{b}__shuffle{r}']) for r in range(10)])
            ax.fill_between(range(1,8),controls.min(0),controls.max(0),alpha=.3,label='10 mismatches: min-max')
            ax.plot(range(1,8),np.diag(values[f'{a}__{b}']),'o-',label='Correct pairing')
            ax.set(title=f'{a} / {b}',xlabel='Sampling slot',ylabel='CKA',ylim=(0,1));ax.legend(fontsize=8)
        fig.savefig(fig_dir/'cka_full_controls.png',dpi=170);plt.close(fig)
    with np.load(FULL/'results/id_mean_curves.npz') as curves:
        fig,axes=plt.subplots(3,2,figsize=(14,12),constrained_layout=True)
        for row,name in enumerate(names):
            for metric_index,title in enumerate(['Insertion: higher AUC preferred','Deletion: lower AUC preferred']):
                ax=axes[row,metric_index]
                for k,method in enumerate(METHODS):
                    ax.plot(FRACTIONS,curves[name][k,metric_index],label=method,
                            color=plt.cm.tab10(k) if k<7 else 'black',linestyle='-' if k<7 else '--')
                ax.set(title=f'{name}: {title}',xlabel='Fraction of pixels modified',ylabel='Mean fixed-target probability (50,000 images)',xlim=(0,1),ylim=(0,1))
                ax.grid(alpha=.2)
                if row==0:ax.legend(ncol=4,fontsize=8)
        fig.savefig(fig_dir/'id_full_curves.png',dpi=170);plt.close(fig)
    table=pd.read_csv(FULL/'results/id_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    for metric,ax in zip(['insertion','deletion'],axes):
        for j,name in enumerate(names):
            t=table[(table.model==name)&(table.method!='random')]
            mean=t[f'{metric}_gain_vs_random'].to_numpy()
            prefix='ins' if metric=='insertion' else 'del'
            low=t[f'{prefix}_gain_ci_low'].to_numpy();high=t[f'{prefix}_gain_ci_high'].to_numpy()
            xs=np.arange(7)+(j-1)*.15
            ax.vlines(xs,low,high,color=plt.cm.tab10(j),linewidth=1.5)
            ax.plot(xs,mean,'o',color=plt.cm.tab10(j),label=name)
        ax.axhline(0,color='black',linestyle='--',linewidth=1)
        ax.set_xticks(range(7),METHODS[:7]);ax.set(title=f'{metric.title()} gain vs random (positive preferred)',xlabel='Sampling slot',ylabel='Mean gain; 95% class-bootstrap interval')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(fig_dir/'id_full_gains.png',dpi=170);plt.close(fig)

def run_full_study(manifest=None):
    started=time.monotonic()
    manifest=prepare_full_dataset() if manifest is None else manifest
    resources=model_resources();write_json(FULL/'run_config.json',{'config':CONFIG,'resources':resources,
        'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,
        'gpu':torch.cuda.get_device_name(0),'started_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')})
    features,metadata=extract_features(manifest,resources)
    run_cka(manifest,features)
    del features;gc.collect();torch.cuda.empty_cache()
    precision=run_insertion_deletion(manifest,resources)
    summary,audit=aggregate_id(manifest)
    make_figures(manifest)
    record={'complete':True,'config':CONFIG,'n_cka':N_FULL,'n_insertion_deletion':N_FULL,
        'n_models':3,'cka_scores':147,'gradcam_combinations':N_FULL*3*7,
        'random_controls':N_FULL*3,'curve_points_including_shared_endpoints':N_FULL*3*8*2*21,
        'data_manifest_sha256':digest(FULL/'data/manifest.json'),'model_resources':resources,
        'precision':precision,'feature_metadata':metadata,'id_audit':audit,
        'results_sha256':{p.name:digest(p) for p in (FULL/'results').glob('*') if p.is_file()},
        'seconds_this_invocation':time.monotonic()-started,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z')}
    write_json(FULL/'complete.json',record)
    progress('complete',N_FULL,N_FULL,'Both CKA and seven-layer insertion/deletion cover all 50,000 validation images')
    return record

def main():
    # Windows releases this byte lock automatically even after a crashed process.
    import msvcrt
    lock=(FULL/'pipeline.lock').open('a+b')
    if lock.tell()==0:lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        lock.close()
        print('Full validation is already running. Follow full_validation_study/progress.json; do not start a second job.',flush=True)
        return
    try:
        run_cli()
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()

def run_cli():
    parser=argparse.ArgumentParser();parser.add_argument('--download-only',action='store_true')
    args=parser.parse_args()
    m=prepare_full_dataset()
    if args.download_only:return
    run_full_study(m)
    # Packaging is separate from the scientific functions embedded in the notebook.
    import subprocess
    subprocess.run([sys.executable,str(ROOT/'build_full_submission_notebooks.py')],check=True,cwd=str(ROOT))

if __name__=='__main__':main()
