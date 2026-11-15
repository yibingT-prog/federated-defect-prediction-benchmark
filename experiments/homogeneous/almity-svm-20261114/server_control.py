"""Server deployment checks and detached launch; does not change frozen experiment files."""
import argparse,json,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['validate','start','status']);a=parser.parse_args()
    os.chdir(ROOT)
    if a.action=='validate':
        for script in ['verify_package.py','check.py']:
            subprocess.run([sys.executable,script],check=True)
        import train
        tasks=json.loads((ROOT/'tasks.json').read_text())
        for target in tasks:train.execute(target,0,smoke=True)
        train.execute('ar5',0)
        subprocess.run([sys.executable,'summarize.py'],check=True)
        report=dict(status='passed',platform=platform.platform(),python=platform.python_version(),smoke_targets=len(tasks),smoke_rounds=2,full_pilot=dict(target='ar5',seed=0,rounds=100),validated_unix=time.time())
        (ROOT/'server_validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    elif a.action=='start':
        assert json.loads((ROOT/'server_validation.json').read_text())['status']=='passed'
        if (ROOT/'scheduler.lock').exists():raise SystemExit('Scheduler lock exists; no second scheduler started')
        if (ROOT/'STOP').exists():raise SystemExit('STOP exists; no scheduler started')
        with (ROOT/'scheduler.log').open('a') as log:
            proc=subprocess.Popen([sys.executable,'-u','schedule.py','--workers','4'],cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (ROOT/'scheduler.pid').write_text(str(proc.pid))
        print(json.dumps(dict(started_pid=proc.pid,workers=4,expected=300)))
    else:
        state=json.loads((ROOT/'status.json').read_text())
        progress=[]
        for t,s in state['active']:
            p=ROOT/'runs'/t/str(s)/'progress.json'
            if p.exists():progress.append(json.loads(p.read_text()))
        print(json.dumps(dict(status=state['status'],completed=len(state['completed']),failed=state['failed'],queued=state['queued'],active=progress,updated_unix=state['updated_unix'])))
        p=ROOT/'verification.json'
        if p.exists():print('VERIFICATION',p.read_text())

if __name__=='__main__':main()
