"""Prepare the existing 5070ti environment's resources without changing packages."""
from pathlib import Path
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import urllib.request
import winreg

import numpy as np
import psutil
import torch
import torchvision
from torchvision import datasets, models

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / 'resource_status.json'
MODEL_DIR = ROOT / '.cache' / 'torch' / 'hub' / 'checkpoints'
DATA_DIR = ROOT / 'data'


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, destination, algorithm, expected):
    """Download from the official source, and publish only a verified file."""
    if destination.exists() and digest(destination, algorithm).startswith(expected):
        print('Verified existing:', destination.name, flush=True)
        return
    temporary = destination.with_name(destination.name + '.partial')
    last_error = None
    for attempt in range(1, 4):
        try:
            print('Downloading:', url, 'attempt', attempt, flush=True)
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = {'User-Agent': 'Mozilla/5.0'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
              resume = response.status == 206 and offset > 0
              if resume and not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                  raise RuntimeError('Unexpected range response')
              count = offset if resume else 0
              total = int(response.headers.get('Content-Length', 0)) + count
              with temporary.open('ab' if resume else 'wb') as output:
                next_log = count + 32 * 1024 * 1024
                start = time.monotonic()
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    output.write(block)
                    count += len(block)
                    if count >= next_log:
                        print(destination.name, round(count / 1024**2), '/',
                              round(total / 1024**2) if total else '?', 'MiB',
                              round(time.monotonic() - start, 1), 'sec', flush=True)
                        next_log = count + 32 * 1024 * 1024
            actual = digest(temporary, algorithm)
            if not actual.startswith(expected):
                temporary.unlink()
                raise RuntimeError('Checksum mismatch for ' + destination.name)
            temporary.replace(destination)
            print('Downloaded and verified:', destination.name, flush=True)
            return
        except Exception as exc:
            last_error = exc
            print('Download error:', repr(exc), flush=True)
    raise RuntimeError('Could not download ' + url) from last_error


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    old_cache = Path(torch.hub.get_dir()) / 'checkpoints'
    torch.hub.set_dir(str(MODEL_DIR.parent))
    torch.set_num_threads(8)
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                        r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
        cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    report = {
        'checked_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'python_executable': sys.executable,
        'python_version': platform.python_version(),
        'torch': torch.__version__,
        'torchvision': torchvision.__version__,
        'cuda_runtime': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'cpu': cpu,
        'cpu_physical_cores': psutil.cpu_count(logical=False),
        'cpu_logical_cores': psutil.cpu_count(),
        'ram_gib': psutil.virtual_memory().total / 1024**3,
        'disk_free_gib': shutil.disk_usage(ROOT).free / 1024**3,
        'models': [],
        'dataset': {},
        'complete': False,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update(gpu=props.name, vram_gib=props.total_memory / 1024**3,
                      gpu_capability=list(torch.cuda.get_device_capability(0)))
    REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('Hardware/environment:', json.dumps(report, ensure_ascii=False), flush=True)
    try:
        for name, builder, weights in [
            ('vgg16', models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
            ('resnet50', models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1),
            ('mobilenet_v2', models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V1),
        ]:
            filename = weights.url.rsplit('/', 1)[1]
            expected = filename.rsplit('-', 1)[1].split('.')[0]
            destination = MODEL_DIR / filename
            cached = old_cache / filename
            if not destination.exists() and cached.exists() and digest(cached).startswith(expected):
                shutil.copyfile(cached, destination)
                print('Copied verified official weight from existing cache:', filename, flush=True)
            download(weights.url, destination, 'sha256', expected)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = builder(weights=weights, progress=False).eval().to(device)
            with torch.no_grad():
                output = model(torch.zeros(1, 3, 224, 224, device=device))
            assert list(output.shape) == [1, 1000] and torch.isfinite(output).all().item()
            report['models'].append({
                'name': name, 'weights': str(weights), 'url': weights.url,
                'path': str(destination.relative_to(ROOT)),
                'bytes': destination.stat().st_size,
                'sha256': digest(destination),
                'smoke_test_device': device, 'output_shape': list(output.shape),
                'finite': True,
            })
            print('Model GPU forward passed:', name, flush=True)
            del model, output
            if device == 'cuda':
                torch.cuda.empty_cache()
            REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')

        report['models_ready'] = True
        if '--models-only' in sys.argv:
            report['dataset'] = {'name': 'ImageNet-1K', 'ready': False,
                                 'status': 'Awaiting authorized ImageNet validation access; CIFAR10 disabled'}
            print('All model weights verified; ImageNet data preparation is separate.', flush=True)
            return
        if '--cifar10' not in sys.argv:
            raise RuntimeError('ImageNet is the selected route. Dataset downloads require explicit selection.')
        archive = DATA_DIR / datasets.CIFAR10.filename
        # Current official site redirects to cave.cs.toronto.edu; use its HTTPS endpoint.
        url = 'https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
        try:
            dataset = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=False)
            print('Existing CIFAR10 data passed torchvision integrity checks', flush=True)
        except RuntimeError:
            download(url, archive, 'md5', datasets.CIFAR10.tgz_md5)
            dataset = datasets.CIFAR10(root=str(DATA_DIR), train=False, download=True)
        assert len(dataset) == 10000
        assert np.bincount(dataset.targets, minlength=10).tolist() == [1000] * 10
        rng = np.random.default_rng(5495)
        labels = np.asarray(dataset.targets)
        indices = []
        for label in range(10):
            indices.extend(rng.choice(np.flatnonzero(labels == label), 100, replace=False).tolist())
        rng.shuffle(indices)
        subset = {
            'dataset': 'CIFAR10', 'split': 'test', 'seed': 5495,
            'n': len(indices), 'per_class': 100,
            'indices': indices,
            'image_ids': [f'cifar10-test-{i:05d}' for i in indices],
            'labels': labels[indices].tolist(), 'classes': dataset.classes,
        }
        subset_path = DATA_DIR / 'cifar10_cka_subset.json'
        subset_path.write_text(json.dumps(subset, indent=2), encoding='utf-8')
        report['dataset'] = {
            'name': 'CIFAR10', 'source': url, 'root': str(DATA_DIR.relative_to(ROOT)),
            'archive_md5': digest(archive, 'md5') if archive.exists() else None,
            'torchvision_integrity_passed': True,
            'test_samples': len(dataset), 'test_class_counts': [1000] * 10,
            'cka_subset': str(subset_path.relative_to(ROOT)),
            'cka_subset_sha256': digest(subset_path),
            'cka_subset_n': len(indices), 'cka_subset_per_class': 100,
            'imagenet_images_downloaded': False,
            'interpretation': 'ImageNet V1 pretrained models evaluated on CIFAR10 inputs',
        }
        report['complete'] = True
        print('CIFAR10 integrity and balanced 1000-image subset passed', flush=True)
    except Exception as exc:
        report['error'] = repr(exc)
        raise
    finally:
        REPORT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print('COMPLETE:', REPORT, flush=True)


if __name__ == '__main__':
    main()
