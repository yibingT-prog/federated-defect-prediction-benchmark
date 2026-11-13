"""Host-side, bounded environment build -> real TFF checks -> authorized FLR run."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
IMAGE = 'flr-tff020:20261111'
NAME = 'flr-homogeneous-20261111'

def status(phase, **fields):
    value = dict(phase=phase,updated_unix=time.time(),**fields)
    tmp = ROOT/'launch_status.tmp'
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    tmp.replace(ROOT/'launch_status.json')
    print(json.dumps(value),flush=True)

def execute(args, filename, timeout):
    with (ROOT/filename).open('w',encoding='utf-8') as log:
        result = subprocess.run(args,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:
        raise RuntimeError(filename+' failed with exit '+str(result.returncode))

def container_prefix(cpus='2',memory='3g'):
    return ['docker','run','--rm','--cpus',cpus,'--memory',memory,
            '--mount','type=bind,source='+str(ROOT)+',target=/work',IMAGE]

def main():
    try:
        # BuildKit shares/caches identical build steps with any in-flight foreground build.
        status('building_TFF_environment')
        execute(['docker','build','--progress','plain','-t',IMAGE,str(ROOT)],'build.log',3600)
        status('checking_native_TFF_and_SGD')
        execute(container_prefix()+['python','smoke_tff.py'],'smoke_tff.log',300)
        status('checking_real_data_two_rounds')
        execute(container_prefix()+['python','train_tff.py','--target','ar5','--seed','0',
                                    '--rounds','2','--pilot'],'pilot.log',600)
        pilot = json.loads((ROOT/'pilot'/'ar5'/'0'/'metrics.json').read_text())
        assert pilot['rounds']==2 and pilot['clients']==4 and pilot['pilot']
        execute(container_prefix()+['python','-m','pip','freeze'],'environment.freeze.txt',120)
        image = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',IMAGE],text=True).strip()
        files = [p for p in ROOT.glob('*.py')]+[ROOT/'protocol.json',ROOT/'tasks.json',ROOT/'requirements-tff.txt']
        fingerprints = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (ROOT/'run_fingerprint.json').write_text(json.dumps(dict(image=image,files=fingerprints),indent=2))
        cfg = json.loads((ROOT/'protocol.json').read_text())
        cfg['status']='TFF_smoke_and_real_data_pilot_passed_formal_run_launched'
        (ROOT/'protocol.json').write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf-8')
        # The formal scheduler has a different output tree from the two-round pilot.
        command = ['docker','run','-d','--name',NAME,'--cpus','4','--memory','4g','--memory-swap','4g',
            '--mount','type=bind,source='+str(ROOT)+',target=/work',IMAGE,'python','schedule.py']
        container = subprocess.check_output(command,text=True,stderr=subprocess.STDOUT).strip()
        status('formal_training_started',container=container,expected_runs=300,
               framework='TensorFlow Federated 0.20.0',tensorflow='2.8.0')
    except Exception as exc:
        (ROOT/'launch_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        status('failed_before_formal_launch',error=str(exc))
        raise

if __name__ == '__main__': main()
