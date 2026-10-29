"""Execute the self-contained notebook with the current 5070ti interpreter."""
import argparse
import json
import os
from pathlib import Path
import queue
import sys
import time

from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--preflight', action='store_true')
parser.add_argument('--notebook', default='A2-cka-study.ipynb')
args = parser.parse_args()
path = ROOT / args.notebook
notebook = json.loads(path.read_text(encoding='utf-8'))
for cell in notebook['cells']:
    if isinstance(cell['source'],list):
        cell['source']=''.join(cell['source'])
selected = {2, 4, 10} if args.preflight else None
connection = ROOT / 'cka_study' / 'kernel_connection.json'
connection.parent.mkdir(exist_ok=True)
km = KernelManager(kernel_name='python3', connection_file=str(connection))
km.kernel_spec.argv = [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}']
km.start_kernel(cwd=str(path.parent), env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
    IPYTHONDIR=str(ROOT / '.cache' / 'ipython'), MPLCONFIGDIR=str(ROOT / '.cache' / 'matplotlib')))
client = km.blocking_client()
client.start_channels()
success = False
try:
    client.wait_for_ready(timeout=90)
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code' or (selected is not None and index not in selected):
            continue
        print('RUN CELL', index, flush=True)
        cell['outputs'] = []
        message_id = client.execute(cell['source'], store_history=True, stop_on_error=True)
        started = time.monotonic()
        failed = False
        while True:
            if time.monotonic() - started > 21600:
                raise TimeoutError('Notebook cell exceeded 6 hours')
            try:
                message = client.get_iopub_msg(timeout=10)
            except queue.Empty:
                continue
            if message.get('parent_header', {}).get('msg_id') != message_id:
                continue
            kind = message['msg_type']
            content = message['content']
            if kind == 'execute_input':
                cell['execution_count'] = content['execution_count']
            elif kind == 'stream':
                cell['outputs'].append({'output_type':'stream', 'name':content['name'], 'text':content['text']})
                print(content['text'], end='', flush=True)
            elif kind in ('display_data', 'execute_result'):
                output = {'output_type':kind, 'data':content['data'], 'metadata':content.get('metadata', {})}
                if kind == 'execute_result':
                    output['execution_count'] = content['execution_count']
                cell['outputs'].append(output)
            elif kind == 'error':
                cell['outputs'].append({'output_type':'error','ename':content['ename'],
                    'evalue':content['evalue'],'traceback':content['traceback']})
                print('CELL ERROR:', content['ename'], content['evalue'], flush=True)
                failed = True
            elif kind == 'clear_output':
                cell['outputs'] = []
            elif kind == 'status' and content['execution_state'] == 'idle':
                break
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
        if failed:
            raise RuntimeError(f'Notebook stopped at cell {index}; see saved output')
        print('CELL PASSED', index, 'seconds', round(time.monotonic()-started, 1), flush=True)
    success = True
finally:
    notebook['metadata']['task_status'] = (
        'Environment, official weights and CKA formula preflight passed; ImageNet experiment pending'
        if success and args.preflight else 'Full experiment executed successfully'
        if success else 'Execution incomplete; see outputs')
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
    client.stop_channels()
    km.shutdown_kernel(now=True)
    if connection.exists():
        connection.unlink()
print('NOTEBOOK RUN COMPLETE', flush=True)
