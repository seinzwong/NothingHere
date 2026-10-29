"""Build a reviewable GitHub snapshot; preserve every validation curve and audit record."""
from pathlib import Path
import hashlib,json,shutil,time,re,sys,importlib.metadata
import numpy as np

ROOT=Path(__file__).resolve().parent
DEST=ROOT/'_publish'
assert (DEST/'.git').is_dir()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()
def copy(path):
    target=DEST/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)

for path in ROOT.glob('*.py'):copy(path)
for name in ['A2-cka-study.ipynb','plan.md','FULL_RUN_STATUS.md','resource_status.json','full_validation_benchmark.json']:
    copy(ROOT/name)
for name in ['dog.png','cat.png']:copy(ROOT/'搭档已完成'/name)
for path in (ROOT/'submit').rglob('*'):
    if path.is_file():copy(path)
full=ROOT/'full_validation_study'
for path in full.rglob('*'):
    if not path.is_file():continue
    rel=path.relative_to(full)
    if rel.parts[0] in ['results','figures','precision','cat_dog_cases'] or path.suffix=='.json' or path.suffix=='.log':
        if '.invalid.' not in path.name:copy(path)
for path in (ROOT/'cka_study').rglob('*'):
    if path.is_file() and path.relative_to(ROOT/'cka_study').parts[0] in ['results','figures','features']:
        copy(path)
for path in (ROOT/'cka_study').glob('*'):
    if path.is_file() and path.suffix in ['.json','.md']:copy(path)
copy(ROOT/'cka_study/imagenet_subset/manifest.json')

evidence=DEST/'evidence';evidence.mkdir(exist_ok=True)
records=[];omitted=[]
fractions=np.rint(np.linspace(0,224*224,21))/(224*224)
for name in ['vgg16','resnet50','mobilenet_v2']:
    folder=full/'insertion_deletion'/name
    output=evidence/'curves'/name;output.mkdir(parents=True,exist_ok=True)
    parts=sorted(folder.glob('part_*.npz'))
    assert len(parts)==782
    fields={key:[] for key in ['curves','targets','labels','indices','fp32_original_prob','all_zero']}
    count=0;batch_start=0;chunk_records=[];max_error=0.
    with np.load(full/'results'/f'{name}_per_image_auc.npz') as values:
        expected_auc=values['auc'];expected_labels=values['labels'];expected_targets=values['targets']
    for number,path in enumerate(parts):
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        actual=sha(path);assert actual==meta['sha256'],str(path)
        size=path.stat().st_size
        omitted.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'bytes':size,'sha256':actual,'reason':'raw native-resolution CAM arrays omitted; all curves exported losslessly'})
        with np.load(path,allow_pickle=False) as data:
            n=len(data['indices']);indices=data['indices']
            assert np.array_equal(indices,np.arange(count,count+n))
            curves=data['curves'].astype(np.float64)
            areas=((curves[...,1:]+curves[...,:-1])*.5*np.diff(fractions)).sum(-1)
            error=float(np.max(np.abs(areas-expected_auc[indices])))
            max_error=max(error,max_error);assert error<1e-12
            assert np.array_equal(data['labels'],expected_labels[indices])
            assert np.array_equal(data['targets'],expected_targets[indices])
            for key in fields:fields[key].append(data[key].copy())
            count+=n
        if (number+1)%64==0 or number+1==len(parts):
            target=output/f'curves_{batch_start:05d}_{count:05d}.npz'
            np.savez_compressed(target,**{key:np.concatenate(value) for key,value in fields.items()},fractions=fractions)
            chunk_records.append({'path':str(target.relative_to(DEST)).replace('\\','/'),'sha256':sha(target),'n':count-batch_start})
            fields={key:[] for key in fields};batch_start=count
    assert count==50000
    records.append({'model':name,'n':count,'original_shards_verified':len(parts),
        'manual_auc_max_error':max_error,'exported_curves':chunk_records})
    print('Exported and verified:',name,count,flush=True)
for path in (full/'features').rglob('*.npy'):
    omitted.append({'path':str(path.relative_to(ROOT)).replace('\\','/'),'bytes':path.stat().st_size,
                    'sha256':sha(path),'reason':'GAP feature cache omitted; metadata hashes and CKA numeric audit included'})
(evidence/'curve_export.json').write_text(json.dumps({'complete':True,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
    'models':records,'n_per_model':50000,'lossless_float32_curves':True,'original_shard_checksums_verified':2346},indent=2),encoding='utf-8')
(evidence/'omitted_artifacts.json').write_text(json.dumps({'artifacts':omitted,'total_bytes':sum(x['bytes'] for x in omitted),
    'note':'These payloads are retained locally, not hosted by this repository. Hashes document provenance, not independent proof of GPU execution.'},indent=2),encoding='utf-8')
packages=['torch','torchvision','numpy','pandas','Pillow','pyarrow','huggingface-hub','jupyter-client','ipykernel','matplotlib','psutil']
(evidence/'environment_versions.json').write_text(json.dumps({'python':sys.version,'packages':{p:importlib.metadata.version(p) for p in packages}},indent=2),encoding='utf-8')
print('Evidence export complete',flush=True)
