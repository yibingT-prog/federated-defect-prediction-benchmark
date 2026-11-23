"""Author-native federated rounds; only data/evaluation/checkpoint adapters are new."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import argparse,hashlib,json,pickle,random
from pathlib import Path
import numpy as np
import pandas as pd
from data import author,load_task
from metrics import classification,effort,chosen_threshold
ROOT=Path(__file__).resolve().parent

def save(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');tmp.replace(p)

def fingerprint(target):
    names=['train.py','data.py','metrics.py','protocol.json','tasks.json','data_manifest.json','vendor/fed_cpdp.py','splits/'+target+'_splits.json']
    return hashlib.sha256(b''.join((ROOT/n).read_bytes() for n in names)).hexdigest()

def f1_threshold(vals):
    y=np.concatenate([v[0] for v in vals]);p=np.concatenate([v[1] for v in vals])
    return max((classification(y,p,float(t))['F1'],-float(t)) for t in np.linspace(.05,.95,37))[1]*-1

def execute(target,seed,smoke=False,limit=None):
    assert os.environ.get('PYTHONHASHSEED')=='0','Launch Python with PYTHONHASHSEED=0'
    cfg=json.loads((ROOT/'protocol.json').read_text());task=json.loads((ROOT/'tasks.json').read_text())[target]
    folder=ROOT/('smoke' if smoke else 'runs')/target/str(seed);folder.mkdir(parents=True,exist_ok=True);fp=fingerprint(target)
    m=author();clients,tx,y,loc,features=load_task(target,m)
    if (folder/'metrics.json').exists():
        old=json.loads((folder/'metrics.json').read_text());assert old['fingerprint']==fp and old['rounds']==m.ROUNDS;print('SKIP completed',target,seed,flush=True);return
    save(folder/'features.json',features)
    random.seed(seed);np.random.seed(seed);rng=np.random.RandomState(seed+777)
    params=m.MLP2(m.HASH_DIM,m.H1_DIM,m.H2_DIM,1,l2=m.L2_WEIGHT_DECAY,seed=seed).get_params()
    state=dict(global_mean=np.zeros(m.H2_DIM,dtype=np.float32),global_var=np.ones(m.H2_DIM,dtype=np.float32),momentum=np.zeros(0,dtype=np.float32));completed=0
    checkpoint=folder/'checkpoint.pkl'
    if checkpoint.exists():
        with checkpoint.open('rb') as f:saved=pickle.load(f)
        assert saved['fingerprint']==fp;params,state,completed=saved['params'],saved['state'],saved['round'];rng.set_state(saved['rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    rounds=1 if smoke else (limit or m.ROUNDS)
    for r in range(completed,rounds):
        if (ROOT/'STOP').exists():save(folder/'progress.json',dict(status='paused',round=r,target=target,seed=seed));return
        # Exactly the upstream round, including its training-loss aggregation input and RNG behavior.
        params,loss=m.federated_round(clients,params,state,rng,base_lr=m.LEARNING_RATE,local_epochs=m.LOCAL_EPOCHS,seed=seed)
        assert all(np.isfinite(p).all() for p in params) and np.isfinite(loss)
        with (folder/'checkpoint.tmp').open('wb') as f:pickle.dump(dict(params=params,state=state,round=r+1,rng=rng.get_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),fingerprint=fp),f)
        (folder/'checkpoint.tmp').replace(checkpoint)
        save(folder/'progress.json',dict(status='running',round=r+1,rounds=m.ROUNDS,target=target,seed=seed,training_loss=loss))
        print(target,seed,'round',r+1,'/',m.ROUNDS,flush=True)
    if smoke or rounds<m.ROUNDS:
        save(folder/'progress.json',dict(status='smoke_passed' if smoke else 'checkpoint_only',round=rounds,target=target,seed=seed,test_scored=False));return
    vals=[(c['vy'],m.model_predict_scores(params,c['VX'])) for c in clients];p=m.model_predict_scores(params,tx);assert np.isfinite(p).all()
    thresholds=dict(study_mcc=chosen_threshold(vals),fixed_05=.5,validation_f1=f1_threshold(vals))
    report=dict(method='FedCPDP-author-native-homogeneous',target=target,family=task['family'],seed=seed,rounds=m.ROUNDS,fingerprint=fp,effort=effort(y,p,loc))
    for mode,t in thresholds.items():report[mode]=dict(threshold=t,**classification(y,p,t))
    pd.DataFrame(dict(row_id=np.arange(len(y)),y_true=y,probability=p,raw_LOC=loc)).to_csv(folder/'predictions.csv',index=False)
    for c,(vy,vp) in zip(clients,vals):pd.DataFrame(dict(row_id=json.loads((ROOT/'splits'/(target+'_splits.json')).read_text())[c['name']]['validation'],y_true=vy,probability=vp)).to_csv(folder/('validation_'+c['name']+'.csv'),index=False)
    save(folder/'metrics.json',report);save(folder/'progress.json',dict(status='completed',round=m.ROUNDS,target=target,seed=seed));print('COMPLETED',target,seed,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--target',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--smoke',action='store_true');ap.add_argument('--max-rounds',type=int);a=ap.parse_args();assert 0<=a.seed<=9;assert a.max_rounds is None or 1<=a.max_rounds<=50;execute(a.target,a.seed,a.smoke,a.max_rounds)
