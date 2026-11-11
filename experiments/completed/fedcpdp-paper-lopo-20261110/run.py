"""Frozen, resumable local FedCPDP experiment; see protocol.json for deviations."""
import os
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import sys, json, time, hashlib, pickle, argparse, subprocess, traceback
from pathlib import Path
ROOT = Path(__file__).resolve().parent

def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8'); tmp.replace(p)

def setup():
    import importlib.util
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, f1_score
    os.chdir(ROOT)
    spec=importlib.util.spec_from_file_location('fed', ROOT/'vendor/fed_cpdp.py')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    for k,v in json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))['model'].items(): setattr(m,k,v)
    return np,pd,StratifiedGroupKFold,roc_auc_score,confusion_matrix,matthews_corrcoef,f1_score,m

def feature_keys(df):
    import pandas as pd
    x=df.drop(columns='defect_label').reindex(sorted(c for c in df if c!='defect_label'),axis=1)
    return pd.util.hash_pandas_object(x,index=False).astype(str).to_numpy()

def encode(df, fit, m):
    import numpy as np
    # Preserve upstream input feature choices; LOC evaluation is separate.
    excluded={'defect_label'}
    effort=m._find_effort_column(df,m.EFFORT_CANDIDATES)
    if effort: excluded.add(effort)
    cols=[c for c in df if c not in excluded and df[c].dtype.kind in 'fiub']
    a=fit[cols].to_numpy(float); mu=np.nanmean(a,axis=0); sd=np.nanstd(a,axis=0); sd[~np.isfinite(sd)|(sd==0)]=1
    z=np.nan_to_num((df[cols].to_numpy(float)-mu)/sd,nan=0.,posinf=0.,neginf=0.)
    out=np.zeros((len(df),m.HASH_DIM),dtype=np.float32)
    for j,c in enumerate(cols): out[:,m._hash_index(c,m.HASH_DIM)]+=z[:,j]
    return out

def split_source(y,groups):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    for tr,va in StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=20261110).split(np.zeros(len(y)),y,groups):
        if len(np.unique(y[tr]))==len(np.unique(y[va]))==2: return tr,va
    raise ValueError('No legal two-class source training/validation fold')

def classification(y,p,t):
    import numpy as np
    from sklearn.metrics import roc_auc_score,confusion_matrix,matthews_corrcoef
    pred=p>=t; tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(ROC_AUC=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        Recall=float(tp/(tp+fn)) if tp+fn else None,
        MCC=float(matthews_corrcoef(y,pred)) if len(np.unique(y))==2 else None,
        G_mean=float(np.sqrt(tp/(tp+fn)*tn/(tn+fp))) if tp+fn and tn+fp else None)

def effort_metrics(y,p,loc):
    import numpy as np
    result={}
    reason=None
    if not np.isfinite(loc).all() or (loc<=0).any(): reason='Missing, non-finite or non-positive raw LOC'
    elif y.sum()==0: reason='No defective entities'
    if reason: return {'reason':reason,**{f'{k}@{b}':None for k in ['CostEffect','Popt'] for b in [5,10,20,40]}}
    def curve(order,b):
        x=np.r_[0,np.cumsum(loc[order])/loc.sum()]; z=np.r_[0,np.cumsum(y[order])/y.sum()]
        xb=np.r_[x[x<b],b]; zb=np.interp(xb,x,z)
        return float(zb[-1]),float(np.trapezoid(zb,xb))
    idx=np.arange(len(y))
    for budget in [5,10,20,40]:
        b=budget/100; ce,area=curve(np.lexsort((idx,-p/loc)),b)
        _,opt=curve(np.lexsort((idx,-y/loc)),b); _,worst=curve(np.lexsort((idx,y/loc)),b)
        result[f'CostEffect@{budget}']=ce; result[f'Popt@{budget}']=(area-worst)/(opt-worst) if abs(opt-worst)>1e-12 else None
    return result

def threshold(vals, mode):
    import numpy as np
    from sklearn.metrics import f1_score
    candidates=[]
    for t in np.linspace(.05,.95,37):
        if mode=='paper_f1': key=(f1_score(np.concatenate([y for y,p in vals]),np.concatenate([p for y,p in vals])>=t,zero_division=0),-t)
        else:
            met=[classification(y,p,t) for y,p in vals]; key=(np.mean([a['MCC'] for a in met]),np.mean([a['Recall'] for a in met]),-t)
        candidates.append((key,float(t)))
    return max(candidates)[1]

def task(target, rounds=50, smoke=False):
    np,pd,SGKF,auc,cm,mcc,f1,m=setup()
    cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    fingerprint=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in ['run.py','protocol.json','manifest.json','vendor/fed_cpdp.py'])).hexdigest()
    for name,rec in manifest.items():
        assert hashlib.sha256((ROOT/'data'/f'{name}.csv').read_bytes()).hexdigest()==rec['sha256'],name
    out=ROOT/('smoke' if smoke else 'runs')/target; out.mkdir(parents=True,exist_ok=True)
    frames={p.stem:pd.read_csv(p) for p in sorted((ROOT/'data').glob('*.csv'))}
    tar=frames[target]; keys=set(feature_keys(tar)); y=tar.defect_label.to_numpy(int)
    # Native project-level target normalization is transductive feature access.
    tx=encode(tar,tar,m); loc=tar[manifest[target]['loc']].to_numpy(float)
    all_scores=[]; all_vals=[]; start=time.time()
    seeds=[42] if smoke else cfg['seeds']
    for seed in seeds:
        folder=out/str(seed); folder.mkdir(exist_ok=True)
        clients=[]; splitlog={}
        for name,df in frames.items():
            if name==target: continue
            g=feature_keys(df); eligible=np.ones(len(df),bool)
            if set(df.columns)==set(tar.columns): eligible=~np.isin(g,list(keys))
            ids=np.flatnonzero(eligible); d=df.iloc[ids]; groups=g[ids]; yy=d.defect_label.to_numpy(int)
            tr,va=split_source(yy,groups)
            assert len(np.unique(yy[tr]))==len(np.unique(yy[va]))==2,name
            assert not set(groups[tr])&set(groups[va])
            clients.append(dict(name=name,X=encode(d.iloc[tr],d.iloc[tr],m),y=yy[tr],VX=encode(d.iloc[va],d.iloc[tr],m),vy=yy[va]))
            splitlog[name]=dict(train=ids[tr].tolist(),validation=ids[va].tolist(),excluded_target_feature_duplicates=np.flatnonzero(~eligible).tolist())
        write_json(folder/'splits.json',splitlog)
        checkpoint=folder/'checkpoint.pkl'
        rng=np.random.RandomState(seed+777)
        params=m.MLP2(m.HASH_DIM,m.H1_DIM,m.H2_DIM,1,l2=m.L2_WEIGHT_DECAY,seed=seed).get_params()
        state=dict(global_mean=np.zeros(m.H2_DIM),global_var=np.ones(m.H2_DIM),momentum=np.zeros(0))
        completed=0
        if checkpoint.exists():
            # Only this run's own trusted local checkpoint is loaded.
            with checkpoint.open('rb') as f: saved=pickle.load(f)
            assert saved['fingerprint']==fingerprint,'Code or configuration changed; start a separate run'
            params,state,completed=saved['params'],saved['state'],saved['round']; rng.set_state(saved['rng'])
        for r in range(completed,rounds):
            if (ROOT/'STOP').exists(): raise InterruptedError('STOP file requested graceful stop')
            deltas=[]; losses=[]; means=[]; variances=[]; weights=[]
            for c in clients:
                delta,_,hm,hv=m.local_train(c,params,state['global_mean'],state['global_var'],m.LEARNING_RATE,m.LOCAL_EPOCHS,seed)
                local=tuple(a+b for a,b in zip(params,delta)); vp=np.clip(m.model_predict_scores(local,c['VX']),1e-12,1-1e-12)
                # Paper calls for validation loss; use explicitly declared unweighted BCE.
                vy=c['vy']; losses.append(float(-(vy*np.log(vp)+(1-vy)*np.log(1-vp)).mean()))
                if m.USE_DP: delta=m.clip_and_add_noise(delta,m.DP_CLIP_NORM,m.DP_NOISE_MULTIPLIER,rng)
                deltas.append(delta); means.append(hm); variances.append(hv); weights.append(len(c['y']))
            agg=m.aggregate(deltas,losses,means,variances,weights,state); params=tuple(a+b for a,b in zip(params,agg))
            state['global_mean']=np.average(means,axis=0,weights=weights); state['global_var']=np.average(variances,axis=0,weights=weights)
            assert all(np.isfinite(a).all() for a in params)
            tmp=checkpoint.with_suffix('.tmp')
            with tmp.open('wb') as f: pickle.dump(dict(params=params,state=state,round=r+1,rng=rng.get_state(),fingerprint=fingerprint),f)
            tmp.replace(checkpoint)
            write_json(out/'progress.json',dict(target=target,seed=seed,round=r+1,rounds=rounds,elapsed_seconds=time.time()-start,status='training'))
            print(f'{target} seed={seed} round={r+1}/{rounds}',flush=True)
        if smoke:
            write_json(out/'smoke_ok.json',dict(finite=True,rounds=rounds,clients=len(clients),target_scores_computed=False)); return
        scores=m.model_predict_scores(params,tx); vals=[(c['vy'],m.model_predict_scores(params,c['VX'])) for c in clients]
        assert np.isfinite(scores).all()
        all_scores.append(scores); all_vals.append(vals)
        pd.DataFrame(dict(row_id=np.arange(len(y)),y_true=y,score=scores,raw_LOC=loc)).to_csv(folder/'predictions.csv',index=False)
        for i,c in enumerate(clients): pd.DataFrame(dict(y_true=vals[i][0],score=vals[i][1])).to_csv(folder/f'validation_{c["name"]}.csv',index=False)
        report={}
        for mode in ['study_mcc','paper_f1']:
            t=threshold(vals,mode); report[mode]=dict(threshold=t,**classification(y,scores,t))
        report['effort']=effort_metrics(y,scores,loc)
        write_json(folder/'metrics.json',report)
    # Ensemble probabilities; source splits are identical across model seeds.
    # Use only records held out by ALL models for ensemble threshold calibration.
    ensemble=np.mean(all_scores,axis=0); common=[]
    for name in frames:
        if name==target: continue
        logs=[json.loads((out/str(s)/'splits.json').read_text()) for s in seeds]
        ids=sorted(set.intersection(*(set(l[name]['validation']) for l in logs)))
        if not ids: continue
        d=frames[name]; yy=d.iloc[ids].defect_label.to_numpy(int)
        if len(np.unique(yy))<2: continue
        pp=[]
        for s,l in zip(seeds,logs):
            with (out/str(s)/'checkpoint.pkl').open('rb') as f: saved=pickle.load(f)
            pp.append(m.model_predict_scores(saved['params'],encode(d.iloc[ids],d.iloc[l[name]['train']],m)))
        common.append((yy,np.mean(pp,axis=0)))
    er={'effort':effort_metrics(y,ensemble,loc),'threshold_validation_projects':len(common)}
    for mode in ['study_mcc','paper_f1']:
        if common:
            t=threshold(common,mode); er[mode]=dict(threshold=t,**classification(y,ensemble,t))
        else: er[mode]={'reason':'No two-class common held-out validation project'}
    pd.DataFrame(dict(row_id=np.arange(len(y)),y_true=y,score=ensemble,raw_LOC=loc)).to_csv(out/'ensemble_predictions.csv',index=False)
    write_json(out/'ensemble_metrics.json',er)
    write_json(out/'progress.json',dict(target=target,status='completed',elapsed_seconds=time.time()-start))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--target'); ap.add_argument('--smoke',action='store_true'); ap.add_argument('--workers',type=int,default=4); args=ap.parse_args()
    if os.environ.get('PYTHONHASHSEED')!='0':
        raise SystemExit(subprocess.call([sys.executable,'-u',__file__,*sys.argv[1:]],env=dict(os.environ,PYTHONHASHSEED='0')))
    if args.target:
        try: task(args.target,1 if args.smoke else 50,args.smoke)
        except Exception:
            traceback.print_exc(); write_json(ROOT/('smoke' if args.smoke else 'runs')/args.target/'failure.json',{'traceback':traceback.format_exc()}); raise
        return
    queue=[p.stem for p in sorted((ROOT/'data').glob('*.csv'))]; active={}; done=[]; failed=[]
    while queue or active:
        if (ROOT/'STOP').exists(): queue=[]
        while queue and len(active)<args.workers:
            name=queue.pop(0); d=ROOT/'runs'/name; d.mkdir(parents=True,exist_ok=True)
            if (d/'ensemble_metrics.json').exists(): done.append(name); continue
            log=(d/'console.log').open('a',encoding='utf-8')
            proc=subprocess.Popen([sys.executable,'-u',__file__,'--target',name],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
            active[name]=(proc,log)
        for name,(p,log) in list(active.items()):
            code=p.poll()
            if code is not None:
                log.close(); (done if code==0 else failed).append(name); del active[name]
        write_json(ROOT/'status.json',dict(status='running' if queue or active else ('finished_with_failures' if failed else 'completed'),active={n:p.pid for n,(p,l) in active.items()},queued=queue,completed=done,failed=failed,updated_unix=time.time()))
        if queue or active: time.sleep(5)

if __name__=='__main__': main()
