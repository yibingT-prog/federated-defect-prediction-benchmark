"""ALMITY aggregation + linear soft-margin SVM reimplementation (NOT author code)."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import argparse,hashlib,json,time,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from prepare import content_hash
from metrics import classification,effort,chosen_threshold
ROOT=Path(__file__).resolve().parent

def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)

def weights(ys):
    counts=np.array([np.bincount(y,minlength=2) for y in ys],dtype=float);n=counts.sum(1);phi=n/n.sum();p=counts/n[:,None]
    alpha=-np.sum(np.where(p>0,p*np.log(np.maximum(p,1e-300)),0),axis=1)/np.log(2)
    minority=int(np.argmin(counts.sum(0)));psi=counts[:,minority]/counts[:,minority].sum();beta=phi*alpha*psi
    if beta.sum()<=0:raise ValueError('Degenerate aggregation; no silent FedAvg fallback')
    return beta/beta.sum(),dict(Phi=phi.tolist(),alpha=alpha.tolist(),Psi=psi.tolist(),beta=beta.tolist(),minority_label=minority,normalized=(beta/beta.sum()).tolist())

def gradient(w,b,x,y,l2):
    signed=2*y-1;active=signed*(x@w+b)<1;z=signed*active
    return l2*w-x.T@z/len(y),-float(z.mean())

def fingerprint(task):
    return hashlib.sha256(b''.join((ROOT/n).read_bytes() for n in ['train.py','prepare.py','metrics.py','protocol.json','tasks.json','data_manifest.json'])+task['prepared_content_sha256'].encode()).hexdigest()

def execute(target,seed,smoke=False,limit=None):
    cfg=json.loads((ROOT/'protocol.json').read_text());task=json.loads((ROOT/'tasks.json').read_text())[target];params=cfg['training'];folder=ROOT/('smoke' if smoke else 'runs')/target/str(seed);folder.mkdir(parents=True,exist_ok=True)
    arrays=np.load(ROOT/'prepared'/(target+'.npz'));assert content_hash(arrays)==task['prepared_content_sha256'];fp=fingerprint(task)
    if (folder/'metrics.json').exists():
        old=json.loads((folder/'metrics.json').read_text());assert old['fingerprint']==fp and old['rounds']==params['rounds'];print('SKIP completed',target,seed,flush=True);return
    clients=[(arrays[f'X_{i}'],arrays[f'y_{i}']) for i in range(len(task['sources']))];agg,attributes=weights([y for x,y in clients]);save(folder/'aggregation.json',dict(clients=task['sources'],**attributes))
    w=np.zeros(clients[0][0].shape[1]);b=0.;completed=0
    ck=folder/'checkpoint.npz'
    if ck.exists():
        with np.load(ck) as state:
            assert str(state['fingerprint'])==fp;w=state['w'].copy();b=float(state['b']);completed=int(state['round'])
    rounds=min(params['rounds'],2 if smoke else (limit or params['rounds']))
    for r in range(completed,rounds):
        if (ROOT/'STOP').exists():save(folder/'progress.json',dict(status='paused',round=r,target=target,seed=seed));return
        local=[]
        for i,(x,y) in enumerate(clients):
            lw=w.copy();lb=b;rng=np.random.default_rng(np.random.SeedSequence([seed,r,i]))
            for e in range(params['local_epochs']):
                order=rng.permutation(len(y))
                for j in range(0,len(y),params['batch_size']):
                    ix=order[j:j+params['batch_size']];gw,gb=gradient(lw,lb,x[ix],y[ix],params['l2']);lw-=params['lr']*gw;lb-=params['lr']*gb
            local.append((lw,lb))
        w=sum(a*v[0] for a,v in zip(agg,local));b=float(sum(a*v[1] for a,v in zip(agg,local)));assert np.isfinite(w).all() and np.isfinite(b)
        with (folder/'checkpoint.tmp').open('wb') as f:np.savez(f,w=w,b=b,round=r+1,fingerprint=fp)
        (folder/'checkpoint.tmp').replace(ck);save(folder/'progress.json',dict(status='running',round=r+1,rounds=params['rounds'],target=target,seed=seed))
        if (r+1)%10==0:print(target,seed,r+1,flush=True)
    if smoke or rounds<params['rounds']:
        save(folder/'progress.json',dict(status='smoke_passed' if smoke else 'checkpoint_only',round=rounds,target=target,seed=seed,test_scored=False));return
    # Evaluation adapter: sigmoid calibration fit ONLY to source-validation margins.
    margins=[arrays[f'V_{i}']@w+b for i in range(len(clients))];labels=[arrays[f'v_{i}'] for i in range(len(clients))]
    calibration=LogisticRegression(C=1.,solver='lbfgs',max_iter=1000,random_state=seed).fit(np.concatenate(margins).reshape(-1,1),np.concatenate(labels))
    ca=float(calibration.coef_[0,0]);cb=float(calibration.intercept_[0]);vals=[(y,expit(ca*z+cb)) for y,z in zip(labels,margins)];threshold=chosen_threshold(vals)
    raw=arrays['X_test']@w+b;p=expit(ca*raw+cb);y=arrays['y_test'];loc=arrays['loc']
    report=dict(method='ALMITY-linear-SVM-paper-reimplementation',target=target,family=task['family'],seed=seed,rounds=rounds,fingerprint=fp,study=dict(threshold=threshold,**classification(y,p,threshold)),fixed_probability=dict(threshold=.5,**classification(y,p,.5)),native_margin=dict(threshold=0.,**classification(y,raw,0.)),effort=effort(y,p,loc),calibration=dict(slope=ca,intercept=cb,fit='pooled source validation only; reused for threshold selection',warning='Calibration is an evaluation adaptation, not a verified original ALMITY component'))
    pd.DataFrame(dict(row_id=np.arange(len(y)),y_true=y,margin=raw,probability=p,raw_LOC=loc)).to_csv(folder/'predictions.csv',index=False)
    for i,n in enumerate(task['sources']):pd.DataFrame(dict(y_true=labels[i],margin=margins[i],probability=vals[i][1])).to_csv(folder/('validation_'+n+'.csv'),index=False)
    save(folder/'metrics.json',report);save(folder/'progress.json',dict(status='completed',round=rounds,target=target,seed=seed));print('COMPLETED',target,seed,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--max-rounds',type=int);a=ap.parse_args();assert 0<=a.seed<=9;execute(a.target,a.seed,a.smoke,a.max_rounds)
if __name__=='__main__':main()
