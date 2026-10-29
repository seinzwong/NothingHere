"""Keep the explicitly requested long run alive and report honest completion."""
from pathlib import Path
import ctypes,json,os,subprocess,sys,time
import psutil

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'full_validation_study'
FOLDER.mkdir(exist_ok=True)
(FOLDER/'supervisor.pid').write_text(str(os.getpid()),encoding='utf-8')
ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # Prevent automatic system sleep for this job only.

def save(record):
    record['updated_at']=time.strftime('%Y-%m-%dT%H:%M:%S%z')
    (FOLDER/'supervisor_status.json').write_text(json.dumps(record,indent=2),encoding='utf-8')

def existing_process():
    p=FOLDER/'run.pid'
    if not p.exists():return None
    try:
        proc=psutil.Process(int(p.read_text().strip()))
        if any('full_validation_pipeline.py' in x for x in proc.cmdline()):return proc
    except (psutil.Error,ValueError):pass
    return None

def published():
    for name in ['A2-cka-study-ZH.ipynb','A2-cka-study-EN.ipynb']:
        p=ROOT/'submit'/name
        if not p.exists():return False
        n=json.loads(p.read_text(encoding='utf-8'))
        if n.get('metadata',{}).get('scope')!='Full ImageNet validation, 50,000 images for both CKA and insertion/deletion':return False
        if any(c.get('execution_count') is None for c in n['cells'] if c['cell_type']=='code'):return False
        if any(o.get('output_type')=='error' for c in n['cells'] for o in c.get('outputs',[])):return False
    return True

try:
    attempts=0
    while attempts<4:
        proc=existing_process()
        if proc is None:
            if published():save({'status':'complete','complete':True});break
            attempts+=1
            with (FOLDER/'run.log').open('a',encoding='utf-8') as output,(FOLDER/'run.err.log').open('a',encoding='utf-8') as errors:
                child=subprocess.Popen([sys.executable,'-u','full_validation_pipeline.py'],cwd=ROOT,
                    stdout=output,stderr=errors,creationflags=subprocess.CREATE_NO_WINDOW)
            (FOLDER/'run.pid').write_text(str(child.pid),encoding='utf-8');proc=psutil.Process(child.pid)
        save({'status':'running','complete':False,'engine_pid':proc.pid,'supervisor_pid':os.getpid(),'restart_attempts':attempts})
        while proc.is_running():
            try:
                if proc.status()==psutil.STATUS_ZOMBIE:break
            except psutil.Error:break
            time.sleep(10)
        if published():save({'status':'complete','complete':True});break
        save({'status':'retry_pending','complete':False,'restart_attempts':attempts})
        time.sleep(30)
    else:
        save({'status':'needs_attention','complete':False,'message':'Repeated failures; inspect run.err.log. No partial results published as full.'})
finally:
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
