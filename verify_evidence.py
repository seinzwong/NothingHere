"""CPU-only independent reaggregation of all published perturbation curves and CIs."""
from pathlib import Path
import hashlib,json,base64,ast
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
FULL=ROOT/'full_validation_study'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def check(condition,message):
    if not condition:raise AssertionError(message)

completion=json.loads((FULL/'complete.json').read_text(encoding='utf-8'))
check(completion['n_cka']==completion['n_insertion_deletion']==50000,'Full-run sample count')
for name,value in completion['results_sha256'].items():check(sha(FULL/'results'/name)==value,'Result SHA: '+name)
manifest_path=FULL/'data/manifest.json'
check(sha(manifest_path)==completion['data_manifest_sha256'],'Image manifest SHA')
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
check(len(manifest['images'])==50000,'Manifest count')
labels=np.array([r['label'] for r in manifest['images']])
check(np.array_equal(np.bincount(labels,minlength=1000),np.full(1000,50)),'Balanced classes')
check(len(pd.read_csv(FULL/'results/cka_main.csv'))==147,'147 CKA combinations')
export=json.loads((ROOT/'evidence/curve_export.json').read_text(encoding='utf-8'))
summary=pd.read_csv(FULL/'results/id_summary.csv')
boot=np.random.default_rng(5495+30000).integers(0,1000,size=(1000,1000))
reports=[]
for model in export['models']:
    name=model['model'];areas=[];zeros=np.zeros(7,dtype=np.int64);count=0;sum_curves=np.zeros((8,2,21))
    with np.load(FULL/'results'/f'{name}_per_image_auc.npz') as data:
        saved_auc=data['auc'];saved_labels=data['labels'];saved_targets=data['targets']
    for chunk in model['exported_curves']:
        path=ROOT/chunk['path'];check(sha(path)==chunk['sha256'],'Curve export SHA')
        with np.load(path,allow_pickle=False) as data:
            c=data['curves'].astype(np.float64);n=len(c);idx=data['indices']
            check(c.shape==(n,8,2,21) and np.isfinite(c).all(),'Curve dimensions/finite')
            check(c.min()>=0 and c.max()<=1+1e-6,'Probability range')
            check(np.array_equal(idx,np.arange(count,count+n)),'Continuous indices')
            check(np.array_equal(data['labels'],labels[idx]),'Manifest class alignment')
            check(np.array_equal(data['targets'],saved_targets[idx]),'Saved targets')
            check(np.allclose(c[:,:,0,0],c[:,:,1,-1],atol=1e-7),'Blank endpoints')
            check(np.allclose(c[:,:,0,-1],c[:,:,1,0],atol=1e-7),'Full endpoints')
            auc=((c[...,1:]+c[...,:-1])*.5*np.diff(data['fractions'])).sum(-1)
            check(np.allclose(auc,saved_auc[idx],rtol=0,atol=1e-12),'All per-image AUCs')
            zero=data['all_zero']
            for i,slot in np.argwhere(zero):check(np.array_equal(c[i,slot],c[i,7]),'Zero CAM random equivalence')
            zeros+=zero.sum(0);sum_curves+=c.sum(0);areas.append(auc);count+=n
    check(count==50000,'All images covered')
    auc=np.concatenate(areas)
    check(np.array_equal(saved_labels,labels),'Label vector')
    class_auc=np.stack([auc[labels==k].mean(0) for k in range(1000)])
    for slot,method in enumerate([f'L{i:02d}' for i in range(1,8)]+['random']):
        row=summary[(summary.model==name)&(summary.method==method)].iloc[0]
        ins=class_auc[:,slot,0]-class_auc[:,7,0];dele=class_auc[:,7,1]-class_auc[:,slot,1]
        ins_ci=np.quantile(ins[boot].mean(1),[.025,.975]);del_ci=np.quantile(dele[boot].mean(1),[.025,.975])
        actual=[auc[:,slot,0].mean(),auc[:,slot,1].mean(),ins.mean(),*ins_ci,dele.mean(),*del_ci]
        expected=[row.insertion_auc_mean,row.deletion_auc_mean,row.insertion_gain_vs_random,
            row.ins_gain_ci_low,row.ins_gain_ci_high,row.deletion_gain_vs_random,row.del_gain_ci_low,row.del_gain_ci_high]
        check(np.allclose(actual,expected,rtol=0,atol=1e-12),'Summary and paired class bootstrap: '+name+'/'+method)
        check(int(row.all_zero_maps)==(int(zeros[slot]) if slot<7 else 0),'Zero count')
    with np.load(FULL/'results/id_mean_curves.npz') as data:check(np.allclose(sum_curves/count,data[name],rtol=0,atol=1e-12),'Mean curves')
    parts=list((FULL/'insertion_deletion'/name).glob('part_*.json'))
    check(len(parts)==782,'All original shard records')
    reports.append({'model':name,'images':count,'all_auc_recomputed':True,'all_summary_and_ci_recomputed':True,'shard_records':len(parts)})
    print('VERIFIED',name,count,flush=True)
for entry in json.loads((ROOT/'submit/submission_validation.json').read_text())['notebooks']:
    path=ROOT/entry['path'].replace('\\','/');check(sha(path)==entry['sha256'],'Notebook SHA')
    notebook=json.loads(path.read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']));check(cell['execution_count'] is not None,'Unexecuted cell')
        check(not any(x.get('output_type')=='error' for x in cell.get('outputs',[])),'Notebook error')
    for image in notebook['metadata']['case_input_data']:
        check(hashlib.sha256(base64.b64decode(image['bytes_base64'])).hexdigest()==image['sha256'],'Embedded input SHA')
report={'passed':True,'models':reports,'cka_scope':'Saved CKA score count, hashes and original numeric audit checked; raw GAP features not distributed, so no new full CKA recomputation.',
    'scope':'Independent CPU reaggregation from exported curves; does not independently replay model inference or prove historical GPU execution.'}
(ROOT/'evidence/independent_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('PASS: all 150,000 image-model curve sets, AUCs, summaries, class bootstrap CIs and notebook checks')
