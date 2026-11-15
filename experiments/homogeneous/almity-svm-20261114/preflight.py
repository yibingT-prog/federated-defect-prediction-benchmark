"""Local acceptance tests: all target smoke jobs and exact interrupted/resumed training."""
import json,platform,shutil,tempfile
from pathlib import Path
import train
import numpy as np
import pandas,scipy,sklearn
from check import main as check_data

ROOT=Path(__file__).resolve().parent

def main():
    check_data();tasks=json.loads((ROOT/'tasks.json').read_text())
    for target in tasks:train.execute(target,0,smoke=True)
    work=ROOT/'test_work';work.mkdir(exist_ok=True)
    trial=Path(tempfile.mkdtemp(prefix='resume-',dir=work))
    for mode in ['continuous','resumed']:
        root=trial/mode;root.mkdir();(root/'prepared').mkdir()
        for name in ['train.py','prepare.py','metrics.py','protocol.json','tasks.json','data_manifest.json']:
            shutil.copy2(ROOT/name,root/name)
        shutil.copy2(ROOT/'prepared/ar5.npz',root/'prepared/ar5.npz')
        train.ROOT=root
        if mode=='resumed':
            train.execute('ar5',0,limit=2)
            assert not (root/'runs/ar5/0/metrics.json').exists()
            assert int(np.load(root/'runs/ar5/0/checkpoint.npz')['round'])==2
        train.execute('ar5',0)
    c=np.load(trial/'continuous/runs/ar5/0/checkpoint.npz');r=np.load(trial/'resumed/runs/ar5/0/checkpoint.npz')
    for name in ['w','b','round','fingerprint']:np.testing.assert_array_equal(c[name],r[name])
    assert (trial/'continuous/runs/ar5/0/predictions.csv').read_bytes()==(trial/'resumed/runs/ar5/0/predictions.csv').read_bytes()
    train.ROOT=ROOT
    # Keep a complete local pilot in the normal output tree for independent verification.
    train.execute('ar5',0)
    report=dict(status='passed',smoke_targets=len(tasks),smoke_rounds=2,full_pilot=dict(target='ar5',seed=0,rounds=100),resume=dict(interruption_after_round=2,final_round=100,weights_bitwise_equal=True,predictions_byte_equal=True),platform=platform.platform(),python=platform.python_version(),packages=dict(numpy=np.__version__,pandas=pandas.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),linux_execution_tested=False)
    train.save(ROOT/'preflight_summary.json',report);print(json.dumps(report))

if __name__=='__main__':main()
