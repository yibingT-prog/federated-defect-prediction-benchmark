"""Run the fixed 300-task matrix; persist per-round checkpoints and bounded workers."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from train import save
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--targets',nargs='+');ap.add_argument('--seeds',type=int,nargs='+');args=ap.parse_args();assert 1<=args.workers<=32
    tasks=json.loads((ROOT/'tasks.json').read_text());targets=sorted(set(args.targets or tasks));seeds=sorted(set(args.seeds if args.seeds is not None else range(10)))
    assert targets and set(targets)<=set(tasks);assert seeds and set(seeds)<=set(range(10));expected=len(targets)*len(seeds)
    lock=ROOT/'scheduler.lock'
    try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    except FileExistsError:raise SystemExit('scheduler.lock exists: verify no other scheduler, then remove stale lock manually')
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        queue=[(t,s) for s in seeds for t in targets];active={};done=[];failed=[]
        while queue or active:
            while queue and len(active)<args.workers and not (ROOT/'STOP').exists():
                t,s=queue.pop(0);folder=ROOT/'runs'/t/str(s);folder.mkdir(parents=True,exist_ok=True)
                log=(folder/'console.log').open('a',encoding='utf-8');p=subprocess.Popen([sys.executable,'-u',str(ROOT/'train.py'),'--target',t,'--seed',str(s)],stdout=log,stderr=subprocess.STDOUT);active[(t,s)]=(p,log)
            for key,(p,log) in list(active.items()):
                if p.poll() is not None:
                    log.close();del active[key]
                    if (ROOT/'runs'/key[0]/str(key[1])/'metrics.json').exists() and p.returncode==0:done.append(key)
                    elif not (ROOT/'STOP').exists():failed.append(key)
            paused=(ROOT/'STOP').exists()
            save(ROOT/'status.json',dict(status='paused' if paused else ('running' if queue or active else ('completed' if not failed else 'completed_with_failures')),completed=done,failed=failed,active=list(active),queued=len(queue),expected=expected,full_matrix_expected=300,updated_unix=time.time()))
            if paused and not active:break
            if queue or active:time.sleep(2)
        if not queue and not active:subprocess.run([sys.executable,str(ROOT/'summarize.py')],check=True)
        if failed:raise SystemExit('Some jobs failed; inspect status.json and per-job console.log')
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
