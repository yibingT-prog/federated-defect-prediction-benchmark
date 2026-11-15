"""Within-family project LOPO, independent client splits; no cross-client grouping."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
ROOT=Path(__file__).resolve().parent

def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def keys(d):
    x=d.drop(columns='defect_label').reindex(sorted(c for c in d if c!='defect_label'),axis=1)
    return pd.util.hash_pandas_object(x,index=False).astype(str).to_numpy()

def content_hash(arrays):
    h=hashlib.sha256()
    for k in sorted(arrays):
        a=np.asarray(arrays[k]);h.update(k.encode());h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes())
    return h.hexdigest()

def scale(fit,x):
    lo=fit.min(0);span=fit.max(0)-lo;span[span==0]=1
    return np.asarray((x-lo)/span,dtype=np.float64)

def main():
    manifest=json.loads((ROOT/'data_manifest.json').read_text());frames={};kk={};tasks={};audit={}
    for n,r in manifest.items():
        p=ROOT/'data'/(n+'.csv');assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],n
        d=pd.read_csv(p);assert np.isfinite(d.to_numpy(float)).all();assert set(d.defect_label.unique())=={0,1};frames[n]=d;kk[n]=keys(d)
    (ROOT/'prepared').mkdir(exist_ok=True)
    for target,d in sorted(frames.items()):
        sources=sorted(n for n in frames if n!=target and manifest[n]['family']==manifest[target]['family']);assert len(sources)>=2
        cols=[c for c in d if c!='defect_label'];assert all(set(frames[n].columns)==set(d.columns) for n in sources)
        x=d[cols].to_numpy(float);arrays=dict(X_test=scale(x,x),y_test=d.defect_label.to_numpy(np.int64),loc=d[manifest[target]['loc']].to_numpy(float));splits={};counts={}
        for i,n in enumerate(sources):
            ids=np.flatnonzero(~np.isin(kk[n],kk[target]));x=frames[n][cols].to_numpy(float);y=frames[n].defect_label.to_numpy(np.int64)
            # Only this client's labels and feature groups determine its validation split.
            chosen=None
            for fold,(tr,va) in enumerate(StratifiedGroupKFold(5,shuffle=True,random_state=20261110).split(np.zeros(len(ids)),y[ids],kk[n][ids])):
                if len(np.unique(y[ids[tr]]))==len(np.unique(y[ids[va]]))==2:chosen=(ids[tr],ids[va],fold);break
            if chosen is None:raise ValueError('No legal local split: '+target+'/'+n)
            it,iv,fold=chosen
            assert not set(kk[n][it])&set(kk[n][iv])
            arrays.update({f'X_{i}':scale(x[it],x[it]),f'V_{i}':scale(x[it],x[iv]),f'y_{i}':y[it],f'v_{i}':y[iv]})
            splits[n]=dict(train=it.tolist(),validation=iv.tolist(),excluded_target_feature_duplicates=np.flatnonzero(np.isin(kk[n],kk[target])).tolist())
            counts[n]=dict(train=len(it),validation=len(iv),train_positives=int(y[it].sum()),validation_positives=int(y[iv].sum()),validation_fraction=len(iv)/(len(it)+len(iv)),fold=fold)
        reference=json.loads((ROOT/'split_reference'/(target+'_splits.json')).read_text())
        assert splits==reference,'Regenerated indices differ from frozen FLR: '+target
        np.savez_compressed(ROOT/'prepared'/(target+'.npz'),**arrays);save(ROOT/'prepared'/(target+'_splits.json'),splits)
        tasks[target]=dict(family=manifest[target]['family'],sources=sources,features=cols,n_test=len(d),split_seed=20261110,prepared_content_sha256=content_hash(arrays))
        audit[target]=dict(within_client_train_validation_overlap=0,source_target_overlap=0,cross_client_train_validation_overlap='not constrained; not part of acceptance criteria',matches_frozen_flr_indices=True,clients=counts)
    save(ROOT/'tasks.json',tasks);save(ROOT/'split_audit.json',dict(status='passed',targets=len(tasks),protocol='rq1-project-lopo-local-groups-v3',global_feature_isolation_claimed=False,targets_detail=audit))
    print('PASS:',len(tasks),'tasks; independent local splits reproduce frozen FLR indices')

if __name__=='__main__':main()
