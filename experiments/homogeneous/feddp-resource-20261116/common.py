import hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

ROOT=Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def torch_save(path,obj):
    path=Path(path); tmp=path.with_suffix('.tmp')
    torch.save(obj,tmp); os.replace(tmp,path)

def fingerprint():
    # Changing any executable/config/data/row split invalidates resume.
    paths=[p for p in ROOT.rglob('*') if p.is_file() and (p.parent==ROOT or p.parent.name in ['splits','data','ecfml_vendor'])
           and p.suffix in ['.py','.json','.csv'] and p.name not in ['package_manifest.json','status.json','verification.json','preflight.json','all_runs.csv','project_summary.csv','family_summary.csv']]
    h=hashlib.sha256()
    for p in sorted(paths,key=lambda p:str(p.relative_to(ROOT)).replace('\\','/')):
        h.update(str(p.relative_to(ROOT)).replace('\\','/').encode()); h.update(p.read_bytes())
    return h.hexdigest()

def load_data(target):
    cfg=read(ROOT/'protocol.json'); tasks=read(ROOT/'tasks.json'); task=tasks[target]
    manifest=read(ROOT/'data_manifest.json'); splits=read(ROOT/'splits'/f'{target}_splits.json')
    names=[target,*task['sources']]+([task['public']] if task['public'] else [])
    frames={}
    for n in names:
        path=ROOT/'data'/f'{n}.csv'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest[n]['sha256'],n
        frames[n]=pd.read_csv(path)
    columns=[c for c in frames[target] if c!='defect_label']
    assert all([c for c in df if c!='defect_label']==columns for df in frames.values())
    scaler_class=StandardScaler if cfg['method']=='ecfml' else MinMaxScaler
    records={}; audit={}; scalers={}
    for n in task['sources']:
        df=frames[n]; tr=np.array(splits[n]['train']); va=np.array(splits[n]['validation'])
        assert not set(tr)&set(va)
        x=df[columns].to_numpy(np.float64); y=df.defect_label.to_numpy(np.int64)
        assert np.isfinite(x).all() and len(np.unique(y[tr]))==2 and len(np.unique(y[va]))==2
        sc=scaler_class().fit(x[tr]); scalers[n]=sc
        records[n]=dict(x=sc.transform(x[tr]).astype('float32'),y=y[tr],vx=sc.transform(x[va]).astype('float32'),vy=y[va],train=tr,validation=va)
        audit[n]=dict(train=tr.tolist(),validation=va.tolist(),scaler_mean=getattr(sc,'mean_',np.array([])).tolist(),scaler_scale=sc.scale_.tolist(),scaler_min=getattr(sc,'min_',np.array([])).tolist())
    df=frames[target]; x=df[columns].to_numpy(float); y=df.defect_label.to_numpy(np.int64)
    if cfg['method']=='ecfml':
        sp=read(ROOT/'target_splits.json')[target]; tr=np.array(sp['support']); te=np.array(sp['query'])
        sc=scaler_class().fit(x[tr]); tx=sc.transform(x[tr]).astype('float32'); ty=y[tr]
    else:
        tr=np.array([],dtype=int); te=np.arange(len(y)); sc=scaler_class().fit(x); tx=None;ty=None
    target_data=dict(x=sc.transform(x[te]).astype('float32'),y=y[te],support_x=tx,support_y=ty,query=te,support=tr,
                     loc=df[manifest[target]['loc']].to_numpy(float)[te],effort_valid=manifest[target]['invalid_loc']==0)
    audit[target]=dict(support=tr.tolist(),query=te.tolist(),scaler_mean=getattr(sc,'mean_',np.array([])).tolist(),scaler_scale=sc.scale_.tolist(),scaler_min=getattr(sc,'min_',np.array([])).tolist())
    public=None
    if task['public']:
        n=task['public']; raw=frames[n][columns].to_numpy(float); ps=MinMaxScaler().fit(raw)
        public=ps.transform(raw).astype('float32')
        ph=set(pd.util.hash_pandas_object(frames[n][columns],index=False).tolist())
        th=set(pd.util.hash_pandas_object(frames[target][columns],index=False).tolist())
        # Do not silently change the agreed no-global-deduplication protocol.
        audit['public']=dict(name=n,n=len(public),labels_used=False,scaler_scale=ps.scale_.tolist(),scaler_min=ps.min_.tolist(),target_duplicate_feature_groups=len(ph&th),
            source_duplicate_feature_groups={s:len(ph&set(pd.util.hash_pandas_object(frames[s][columns].iloc[records[s]['train']],index=False).tolist())) for s in task['sources']})
    audit['feature_columns']=columns
    return cfg,task,records,target_data,public,audit

def predict(model,x):
    model.eval()
    with torch.no_grad():
        p=torch.cat([torch.softmax(model(torch.from_numpy(x[i:i+512])),dim=1)[:,1] for i in range(0,len(x),512)]).numpy()
    assert np.isfinite(p).all()
    return p
