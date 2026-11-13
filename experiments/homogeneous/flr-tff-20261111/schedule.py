"""Bounded native-TFF jobs. Resume only completed jobs; restart interrupted seeds."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
def save(obj):
    tmp = ROOT/'status.tmp'
    tmp.write_text(json.dumps(obj,indent=2),encoding='utf-8')
    tmp.replace(ROOT/'status.json')

def main():
    tasks = json.loads((ROOT/'tasks.json').read_text())
    cfg = json.loads((ROOT/'protocol.json').read_text())
    # Small tasks first gives early completed validation; all tasks are predeclared.
    targets = sorted(tasks, key=lambda t: (tasks[t]['n_train'],t))
    pending = [(t,s) for s in cfg['training']['seeds'] for t in targets]
    active = {}; completed = []; failed = []
    while pending or active:
        while pending and len(active)<2:
            t,s = pending.pop(0)
            folder = ROOT/'runs'/t/str(s)
            folder.mkdir(parents=True,exist_ok=True)
            if (folder/'metrics.json').exists():
                completed.append([t,s]); continue
            log = (folder/'console.log').open('w',encoding='utf-8')
            env = os.environ.copy(); env['PYTHONHASHSEED']=str(s)
            p = subprocess.Popen([sys.executable,'-u',str(ROOT/'train_tff.py'),
                '--target',t,'--seed',str(s)],stdout=log,stderr=subprocess.STDOUT,env=env)
            active[(t,s)] = (p,log)
        for key,(p,log) in list(active.items()):
            code = p.poll()
            if code is not None:
                log.close(); del active[key]
                valid = (ROOT/'runs'/key[0]/str(key[1])/'metrics.json').exists()
                (completed if code==0 and valid else failed).append(list(key))
        save(dict(status='running' if pending or active else ('completed' if not failed else 'completed_with_failures'),
            expected_runs=300,completed=completed,failed=failed,active=[list(k) for k in active],
            queued=len(pending),updated_unix=time.time()))
        if pending or active: time.sleep(10)
    subprocess.run([sys.executable,str(ROOT/'summarize.py')],check=True)
    print('Finished',len(completed),'failed',len(failed),flush=True)

if __name__ == '__main__': main()
