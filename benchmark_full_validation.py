from pathlib import Path
import json,time,copy
import numpy as np
import torch
from torchvision import models
from PIL import Image

ROOT=Path(__file__).resolve().parent
torch.hub.set_dir(str(ROOT/'.cache/torch/hub'))
torch.set_num_threads(8)
torch.backends.cudnn.benchmark=True
torch.use_deterministic_algorithms(False)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
manifest=json.loads((ROOT/'cka_study/imagenet_subset/manifest.json').read_text(encoding='utf-8'))
transform=models.VGG16_Weights.IMAGENET1K_V1.transforms()
images=[]
for r in manifest['images'][:16]:
    with Image.open(ROOT/r['path']) as im:images.append(transform(im.convert('RGB')))
x=torch.stack(images).to('cuda',memory_format=torch.channels_last)
mask=torch.rand((16,1,224,224),device='cuda')>.5
x=torch.cat([x,x*mask],0).repeat(4,1,1,1).contiguous(memory_format=torch.channels_last)
results=[]
for name,builder,w in [('vgg16',models.vgg16,models.VGG16_Weights.IMAGENET1K_V1),
                       ('resnet50',models.resnet50,models.ResNet50_Weights.IMAGENET1K_V1),
                       ('mobilenet_v2',models.mobilenet_v2,models.MobileNet_V2_Weights.IMAGENET1K_V1)]:
    m=builder(weights=w).eval().to('cuda',memory_format=torch.channels_last)
    with torch.inference_mode():
        ref=m(x).float();target=ref.argmax(1);p_ref=ref.softmax(1).gather(1,target[:,None])
        for dtype in [torch.float32,torch.float16,torch.bfloat16]:
            if dtype!=torch.float32:m=m.to(dtype=dtype)
            inputs=x.to(dtype=dtype)
            for _ in range(2):out=m(inputs)
            torch.cuda.synchronize();start=time.monotonic()
            for _ in range(5):out=m(inputs)
            torch.cuda.synchronize();seconds=time.monotonic()-start
            p=out.float().softmax(1).gather(1,target[:,None])
            row={'model':name,'dtype':str(dtype),'batch':len(x),'images_per_second':len(x)*5/seconds,
                 'max_fixed_target_prob_error':float((p-p_ref).abs().max().item()),
                 'mean_fixed_target_prob_error':float((p-p_ref).abs().mean().item()),
                 'top1_agreement':float((out.argmax(1)==target).float().mean().item()),
                 'peak_vram_gib':torch.cuda.max_memory_allocated()/1024**3}
            print(json.dumps(row),flush=True);results.append(row)
    del m,ref,out,p,p_ref,inputs
    torch.cuda.empty_cache()
(ROOT/'full_validation_benchmark.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
