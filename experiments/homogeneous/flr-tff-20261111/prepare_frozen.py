"""Rebuild executed FLR inputs from frozen indices, without rerunning FedCPDP."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent

def digest(arrays):
    h=hashlib.sha256()
    for k in sorted(arrays):
        a=np.asarray(arrays[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--archive',type=Path);g.add_argument('--data-dir',type=Path);args=ap.parse_args()
    manifest=json.loads((ROOT/'data_manifest.json').read_text());tasks=json.loads((ROOT/'tasks.json').read_text());expected=json.loads((ROOT/'prepared_content_hashes.json').read_text());frames={}
    if args.archive:
        payload=args.archive.read_bytes()
        assert hashlib.sha256(payload).hexdigest()=='4c19cbcf4a8d22fa99f47ec440b99dc716e6148a962d5af4f6f65292137b7801'
        archive=zipfile.ZipFile(io.BytesIO(payload))
    for n,rec in manifest.items():
        if args.archive:
            matches=[x for x in archive.namelist() if Path(x).name==n+'.csv'];assert len(matches)==1
            raw=archive.read(matches[0])
        else:raw=(args.data_dir/(n+'.csv')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==rec['sha256'],n
        frames[n]=pd.read_csv(io.BytesIO(raw))
    def scale(fit,x):
        low=fit.min(axis=0);span=fit.max(axis=0)-low;span[span==0]=1
        return np.asarray((x-low)/span,dtype=np.float32)
    prepared=ROOT/'prepared';prepared.mkdir(exist_ok=True)
    for target,task in tasks.items():
        d=frames[target];x=d[task['features']].to_numpy(float)
        arrays=dict(X_test=scale(x,x),y_test=d.defect_label.to_numpy(int),loc=d[manifest[target]['loc']].to_numpy(float))
        split=json.loads((ROOT/'splits'/(target+'_splits.json')).read_text())
        for i,n in enumerate(task['sources']):
            tr=split[n]['train'];va=split[n]['validation'];x=frames[n][task['features']].to_numpy(float);y=frames[n].defect_label.to_numpy(int)
            arrays.update({f'X_{i}':scale(x[tr],x[tr]),f'V_{i}':scale(x[tr],x[va]),f'y_{i}':y[tr],f'v_{i}':y[va]})
        assert digest(arrays)==expected[target],target
        np.savez_compressed(prepared/(target+'.npz'),**arrays)
        shutil.copy2(ROOT/'splits'/(target+'_splits.json'),prepared/(target+'_splits.json'))
    print('PASS: all 30 prepared tasks match executed array hashes')

if __name__=='__main__':main()
