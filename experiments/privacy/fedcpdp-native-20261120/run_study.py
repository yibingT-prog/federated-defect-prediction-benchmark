"""Pilot gate, then resumable full sweep. Run with PYTHONHASHSEED=0."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ.update(PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
def save(value):
    p=ROOT/'study_status.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--pilot-only',action='store_true');a=ap.parse_args()
    assert 1<=a.workers<=4
    lock=ROOT/'study.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        subprocess.run([sys.executable,str(ROOT/'validate_package.py')],check=True)
        conditions=json.loads((ROOT/'study.json').read_text())['conditions']
        for stage in (['pilot'] if a.pilot_only else ['pilot','formal']):
            for condition in conditions:
                folder=ROOT/'conditions'/condition
                if (folder/'STOP').exists():raise RuntimeError('STOP exists: '+condition)
                save(dict(state='running',stage=stage,condition=condition,updated_unix=time.time()))
                cmd=[sys.executable,'-u','schedule.py','--workers',str(a.workers)]
                if stage=='pilot':cmd+=['--targets','ar1','--seeds','0']
                subprocess.run(cmd,cwd=folder,check=True)
                verified=json.loads((folder/'verification.json').read_text())['verified_runs']
                assert verified>=(1 if stage=='pilot' else 300)
                pilot=json.loads((folder/'runs/ar1/0/metrics.json').read_text());assert pilot['rounds']==50
            if stage=='pilot':
                (ROOT/'pilot_gate.json').write_text(json.dumps({'passed':True,'conditions':conditions,'rounds':50,'test_metrics_used_for_config_selection':False},indent=2))
        save(dict(state='pilot_completed' if a.pilot_only else 'completed',expected=6 if a.pilot_only else 1800,updated_unix=time.time()))
    except BaseException as e:
        save(dict(state='failed',error=str(e),updated_unix=time.time()));raise
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
