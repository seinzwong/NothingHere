"""Two illustrative inputs, kept separate from the completed 50,000-image benchmark."""
import base64
import io


def case_letterbox(image):
    image=image.convert('RGB')
    ratio=min(224/image.width,224/image.height)
    size=(round(image.width*ratio),round(image.height*ratio))
    canvas=Image.new('RGB',(224,224),(124,116,104))
    resized=image.resize(size,Image.Resampling.BILINEAR)
    offset=((224-size[0])//2,(224-size[1])//2)
    canvas.paste(resized,offset)
    pixels=np.asarray(canvas).copy()
    tensor=torch.from_numpy(pixels).permute(2,0,1).float()/255
    tensor=(tensor-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
    return canvas,tensor,{'original_size':list(image.size),'resized_size':list(size),'offset':list(offset)}


def run_cat_dog_cases(case_inputs):
    folder=FULL/'cat_dog_cases';folder.mkdir(exist_ok=True)
    inputs=[];images=[];tensors=[]
    for item in case_inputs:
        raw=base64.b64decode(item['bytes_base64'])
        assert hashlib.sha256(raw).hexdigest()==item['sha256']
        original=folder/(item['id']+'_original.png');original.write_bytes(raw)
        with Image.open(io.BytesIO(raw)) as im:canvas,tensor,geometry=case_letterbox(im)
        canvas.save(folder/(item['id']+'_224.png'))
        inputs.append({'id':item['id'],'sha256':item['sha256'],**geometry})
        images.append(canvas);tensors.append(tensor)
    x=torch.stack(tensors).to(DEVICE,memory_format=torch.channels_last)
    rows=[];model_records=[];started=time.monotonic()
    # Small case study uses FP32 throughout, independently of full-run FP16 gates.
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True,warn_only=True)
    for name,(builder,weights,layers) in SPECS.items():
        model=builder(weights=weights,progress=False).eval().to(DEVICE,memory_format=torch.channels_last)
        for parameter in model.parameters():parameter.requires_grad_(False)
        for module in model.modules():
            if isinstance(module,(torch.nn.ReLU,torch.nn.ReLU6)):module.inplace=False
        with warnings.catch_warnings(record=True) as logged:
            warnings.simplefilter('always')
            cams,targets,reference=collect_cams(model,x,layers)
        ranks,zero=rank_maps(cams,[1000000,1000001])
        curves=evaluate_curves(model,x,ranks,targets).cpu().numpy()
        assert np.max(np.abs(curves[:,:,0,-1]-reference.cpu().numpy()[:,None]))<1e-5
        assert np.allclose(curves[:,:,0,0],curves[:,:,1,-1],atol=1e-7)
        assert np.allclose(curves[:,:,0,-1],curves[:,:,1,0],atol=1e-7)
        for i,slot in torch.nonzero(zero,as_tuple=False).cpu().tolist():
            assert torch.equal(ranks[i,slot],ranks[i,7])
            assert np.allclose(curves[i,slot],curves[i,7],atol=1e-7)
        auc=np.trapezoid(curves.astype(np.float64),x=FRACTIONS,axis=-1)
        manual=((curves[...,1:].astype(np.float64)+curves[...,:-1])*np.diff(FRACTIONS)/2).sum(-1)
        assert np.max(np.abs(auc-manual))<1e-12
        up=torch.stack([F.interpolate(c[:,None],size=(224,224),mode='bilinear',align_corners=False)[:,0] for c in cams],dim=1)
        arrays={'curves':curves,'targets':targets.cpu().numpy(),'original_probability':reference.cpu().numpy(),
                'cams_224':up.cpu().numpy(),'all_zero':zero.cpu().numpy(),'fractions':FRACTIONS}
        for slot,cam in enumerate(cams):arrays[f'raw_cam_{slot+1:02d}']=cam.cpu().numpy()
        path=folder/(name+'.npz');np.savez_compressed(path,**arrays)
        for i,item in enumerate(inputs):
            for slot,method in enumerate(METHODS):
                rows.append({'image':item['id'],'model':name,'method':method,
                    'target_id':int(targets[i]),'target':weights.meta['categories'][int(targets[i])],
                    'original_probability':float(reference[i]),'insertion_auc':float(auc[i,slot,0]),
                    'deletion_auc':float(auc[i,slot,1]),'insertion_gain_vs_random':float(auc[i,slot,0]-auc[i,7,0]),
                    'deletion_gain_vs_random':float(auc[i,7,1]-auc[i,slot,1]),
                    'all_zero':bool(zero[i,slot]) if slot<7 else False})
        model_records.append({'model':name,'sha256':digest(path),'targets':targets.cpu().tolist(),
            'original_probability':reference.cpu().tolist(),'zero_maps':int(zero.sum()),'precision':'FP32',
            'layers':layers,'weights':str(weights),
            'backward_warnings':sorted(set(str(w.message).split(' (')[0] for w in logged))})
        print('Case study completed:',name,flush=True)
        del model,cams,ranks,zero,up,arrays;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(folder/'summary.csv',index=False)
    write_json(folder/'complete.json',{'complete':True,'inputs':inputs,'models':model_records,
        'n_images':2,'gradcam_combinations':42,'random_controls':6,'steps':21,'fixed_original_top1':True,
        'preprocessing':'bilinear aspect-preserving resize, centered letterbox 224, RGB(124,116,104), ImageNet normalization',
        'baseline':'zero normalized tensor (ImageNet mean RGB)','seed':SEED,'random_row_ids':[1000000,1000001],
        'included_in_full_validation_statistics':False,'two_sample_cka_computed':False,
        'seconds':time.monotonic()-started,'finished_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'summary_sha256':digest(folder/'summary.csv'),
        'audit':'fixed endpoints, zero-map random equivalence, finite curves and manual trapezoidal AUC checked'})
    plot_cat_dog_cases()


def plot_cat_dog_cases():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=FULL/'cat_dog_cases'
    table=pd.read_csv(folder/'summary.csv')
    images=[np.array(Image.open(folder/(key+'_224.png')).convert('RGB')) for key in ['dog','cat']]
    fig,axes=plt.subplots(1,2,figsize=(7,4),constrained_layout=True)
    for i,key in enumerate(['dog','cat']):
        axes[i].imshow(images[i]);axes[i].set_title(key.title()+' | letterbox 224 x 224');axes[i].axis('off')
    fig.savefig(folder/'case_inputs.png',dpi=160);plt.close(fig)
    for i,key in enumerate(['dog','cat']):
        fig,axes=plt.subplots(3,8,figsize=(16,7),constrained_layout=True)
        for row,name in enumerate(SPECS):
            record=table[(table.image==key)&(table.model==name)].iloc[0]
            axes[row,0].imshow(images[i]);axes[row,0].set_title(name+'\n'+record.target,fontsize=9)
            with np.load(folder/(name+'.npz')) as data:
                for slot in range(7):
                    cam=data['cams_224'][i,slot];peak=float(cam.max())
                    ax=axes[row,slot+1];ax.imshow(images[i]);ax.imshow(cam/max(peak,1e-12),cmap='jet',alpha=.4,vmin=0,vmax=1)
                    ax.set_title(f'L{slot+1:02d}'+(' | zero' if peak==0 else ''),fontsize=10)
            for ax in axes[row]:ax.axis('off')
        fig.suptitle(key.title()+' | fixed original Top-1 Grad-CAM | each map independently normalized')
        fig.savefig(folder/(key+'_cams.png'),dpi=160);plt.close(fig)
        fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
        for row,name in enumerate(SPECS):
            with np.load(folder/(name+'.npz')) as data:curves=data['curves'][i]
            for metric in range(2):
                ax=axes[row,metric]
                for slot,method in enumerate(METHODS):
                    ax.plot(FRACTIONS,curves[slot,metric],label=method,
                        color='black' if slot==7 else plt.cm.tab10(slot),ls='--' if slot==7 else '-',lw=1.8)
                ax.set_title(name+' | '+('Insertion (higher AUC better)' if metric==0 else 'Deletion (lower AUC better)'))
                ax.set_xlabel('Fraction restored' if metric==0 else 'Fraction removed')
                ax.set_ylabel('Fixed target probability');ax.grid(alpha=.2);ax.legend(ncol=4,fontsize=8)
        fig.suptitle(key.title()+' | 7 Grad-CAM rankings and one seeded random control')
        fig.savefig(folder/(key+'_curves.png'),dpi=150);plt.close(fig)


# CLI
if __name__=='__main__':
    from full_validation_pipeline import *
    case_inputs=[]
    for key in ['dog','cat']:
        path=ROOT/'搭档已完成'/(key+'.png')
        case_inputs.append({'id':key,'sha256':digest(path),'bytes_base64':base64.b64encode(path.read_bytes()).decode('ascii')})
    run_cat_dog_cases(case_inputs)
