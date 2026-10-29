import json
from pathlib import Path
import psutil

for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    if 'python' in (proc.info['name'] or '').lower():
        print(proc.info)
bad = []
files = list(Path('full_validation_study').rglob('*.json'))
for path in files:
    try:
        json.loads(path.read_text(encoding='utf-8'))
    except Exception as error:
        bad.append((str(path), path.stat().st_size, str(error)))
print('INVALID JSON:', bad)
print('Checked:', len(files))
