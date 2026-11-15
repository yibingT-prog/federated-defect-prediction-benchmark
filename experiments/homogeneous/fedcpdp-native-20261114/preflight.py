"""Audit frozen rows and native features; check exact upstream training equivalence."""
import os
os.environ.update(PYTHONHASHSEED='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,pickle,subprocess,sys,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from data import author,load_task,encode
from train import save
ROOT=Path(__file__).resolve().parent

def main():
    m=author();tasks=json.loads((ROOT/'tasks.json').read_text());manifest=json.loads((ROOT/'data_manifest.json').read_text())
    frames={n:pd.read_csv(ROOT/'data'/(n+'.csv')) for n in manifest}
    keys={n:pd.util.hash_pandas_object(d.drop(columns='defect_label').reindex(sorted(c for c in d if c!='defect_label'),axis=1),index=False).astype(str).to_numpy() for n,d in frames.items()}
    for target,t in tasks.items():
        assert set(t['sources'])=={n for n in manifest if n!=target and manifest[n]['family']==t['family']}
        splits=json.loads((ROOT/'splits'/(target+'_splits.json')).read_text())
        for n,s in splits.items():
            tr,va,ex=s['train'],s['validation'],s['excluded_target_feature_duplicates']
            assert not set(tr)&set(va);assert set(tr)|set(va)|set(ex)==set(range(len(frames[n])));assert not (set(tr)|set(va))&set(ex)
            assert not set(keys[n][tr])&set(keys[n][va]);assert set(ex)==set(np.flatnonzero(np.isin(keys[n],keys[target])))
        clients,tx,y,loc,features=load_task(target,m)
        for c in clients:
            train=frames[c['name']].iloc[splits[c['name']]['train']]
            native,_,_=m.hash_numeric_features(train,'defect_label',m.HASH_DIM,True);np.testing.assert_array_equal(c['X'],native)
            assert set(np.unique(c['y']))==set(np.unique(c['vy']))=={0,1}
        native,_,_=m.hash_numeric_features(frames[target],'defect_label',m.HASH_DIM,True);np.testing.assert_array_equal(tx,native)
    save(ROOT/'resolved_author_settings.json',{k:v for k,v in vars(m).items() if k.isupper() and isinstance(v,(str,int,float,bool,list))})
    print('PASS all30 frozen tasks and native feature equivalence',flush=True)
    for target in ['ar5','Apache','CM1','EQ','ant-1.7']:
        subprocess.run([sys.executable,'train.py','--target',target,'--seed','0','--smoke'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'train.py','--target','ar5','--seed','0','--max-rounds','2'],cwd=ROOT,check=True)
    clients,*_=load_task('ar5',m);m.ROUNDS=2;native,_=m.train_one_seed(clients,0)
    with (ROOT/'runs/ar5/0/checkpoint.pkl').open('rb') as f:s=pickle.load(f)
    for a,b in zip(native,s['params']):np.testing.assert_array_equal(a,b)
    subprocess.run([sys.executable,'train.py','--target','ar5','--seed','0'],cwd=ROOT,check=True)
    m.ROUNDS=50;native,_=m.train_one_seed(clients,0)
    with (ROOT/'runs/ar5/0/checkpoint.pkl').open('rb') as f:s=pickle.load(f)
    for a,b in zip(native,s['params']):np.testing.assert_array_equal(a,b)
    subprocess.run([sys.executable,'summarize.py'],cwd=ROOT,check=True)
    save(ROOT/'preflight.json',dict(status='passed',targets_audited=30,smoke_targets=['ar5','Apache','CM1','EQ','ant-1.7'],native_feature_arrays_exact=True,author_core_sha256=hashlib.sha256((ROOT/'vendor/fed_cpdp.py').read_bytes()).hexdigest(),native_training_params_bitwise_equal_at_round2=True,resumed_round50_equals_author_uninterrupted=True,full_pilot=dict(target='ar5',seed=0,rounds=50),global_cross_client_feature_isolation_required=False))
    print('PREFLIGHT PASSED',flush=True)

if __name__=='__main__':main()
