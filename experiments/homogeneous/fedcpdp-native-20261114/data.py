"""Frozen project/row protocol with native FedCPDP feature processing."""
import hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def author():
    cfg=json.loads((ROOT/'protocol.json').read_text())
    path=ROOT/'vendor/fed_cpdp.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['vendor_sha256']
    spec=importlib.util.spec_from_file_location('author_fedcpdp',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for key,value in cfg['model'].items():setattr(m,key,value)
    return m

def encode(df,fit,m):
    exclude=['defect_label'];loc=m._find_effort_column(df,m.EFFORT_CANDIDATES)
    if loc is not None:exclude.append(loc)
    cols=m._numeric_cols(df,exclude);out=np.zeros((len(df),m.HASH_DIM),dtype=np.float32)
    for c in cols:
        a=fit[c].to_numpy(float);mu=np.nanmean(a);sd=np.nanstd(a)
        if sd==0 or np.isnan(sd):sd=1.
        z=np.nan_to_num((df[c].to_numpy(float)-mu)/sd,nan=0.,posinf=0.,neginf=0.)
        out[:,m._hash_index(c,m.HASH_DIM)]+=z
    return out,dict(columns=cols,excluded_effort_column=loc,hash_indices={c:m._hash_index(c,m.HASH_DIM) for c in cols})

def load_task(target,m):
    manifest=json.loads((ROOT/'data_manifest.json').read_text());task=json.loads((ROOT/'tasks.json').read_text())[target]
    splits=json.loads((ROOT/'splits'/(target+'_splits.json')).read_text());frames={}
    for n in [target,*task['sources']]:
        path=ROOT/'data'/(n+'.csv');assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[n]['sha256'];frames[n]=pd.read_csv(path)
    clients=[];features={}
    for n in task['sources']:
        df=frames[n];tr=splits[n]['train'];va=splits[n]['validation'];x,features[n]=encode(df.iloc[tr],df.iloc[tr],m);vx,_=encode(df.iloc[va],df.iloc[tr],m)
        clients.append(dict(name=n,X=x,y=df.defect_label.to_numpy(int)[tr],VX=vx,vy=df.defect_label.to_numpy(int)[va]))
    d=frames[target];x,features[target]=encode(d,d,m)
    return clients,x,d.defect_label.to_numpy(int),d[manifest[target]['loc']].to_numpy(float),features
