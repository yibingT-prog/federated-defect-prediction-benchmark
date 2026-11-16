import argparse,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from common import ROOT,read,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);args=ap.parse_args()
    assert 1<=args.workers<=8
    lock=ROOT/'scheduler.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        targets=sorted(read(ROOT/'tasks.json'));cfg=read(ROOT/'protocol.json')
        queue=[(t,s) for s in cfg['seeds'] for t in targets];active={};done=[];failed=[]
        while queue or active:
            while queue and len(active)<args.workers and not (ROOT/'STOP').exists():
                t,s=queue.pop(0);folder=ROOT/'runs'/t/str(s);folder.mkdir(parents=True,exist_ok=True)
                if (folder/'metrics.json').exists():done.append((t,s));continue
                log=(folder/'console.log').open('a',encoding='utf-8')
                p=subprocess.Popen([sys.executable,'-u',str(ROOT/'train.py'),'--target',t,'--seed',str(s)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                active[(t,s)]=(p,log)
            for key,(p,log) in list(active.items()):
                if p.poll() is not None:
                    log.close();del active[key]
                    if p.returncode==0 and (ROOT/'runs'/key[0]/str(key[1])/'metrics.json').exists():done.append(key)
                    else:failed.append(key)
            # Stop scheduling on failure so a systematic failure does not exhaust the matrix.
            state='failed' if failed else ('paused' if (ROOT/'STOP').exists() else 'running' if queue or active else 'training_complete')
            save(ROOT/'status.json',dict(status=state,completed=done,failed=failed,active=list(active),queued=len(queue),expected=cfg['expected_runs'],updated_unix=time.time()))
            if failed or ((ROOT/'STOP').exists() and not active):break
            if queue or active:time.sleep(2)
        if failed:
            for p,log in active.values():p.wait();log.close()
            raise SystemExit('A task failed; inspect console.log. Remaining matrix was not started.')
        if not queue and not active:
            subprocess.run([sys.executable,str(ROOT/'verify.py')],cwd=ROOT,check=True)
            save(ROOT/'status.json',dict(status='completed_verified',completed=done,failed=[],active=[],queued=0,expected=cfg['expected_runs'],updated_unix=time.time()))
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':main()
