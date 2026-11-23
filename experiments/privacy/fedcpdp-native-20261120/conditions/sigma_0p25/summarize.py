"""Recompute predictions/metrics from trusted local checkpoints and summarize by project."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,pickle
from pathlib import Path
import numpy as np
import pandas as pd
from data import author,load_task
from train import save,fingerprint,f1_threshold
from metrics import classification,effort,chosen_threshold
ROOT=Path(__file__).resolve().parent

def main():
    assert os.environ.get('PYTHONHASHSEED')=='0'
    tasks=json.loads((ROOT/'tasks.json').read_text());rows=[];verified=[];m=author()
    for target,task in tasks.items():
        files=sorted((ROOT/'runs'/target).glob('*/metrics.json'))
        if not files:continue
        clients,tx,y,loc,_=load_task(target,m)
        for file in files:
            r=json.loads(file.read_text());assert r['target']==target and str(r['seed'])==file.parent.name and 0<=r['seed']<=9;assert r['rounds']==50 and r['fingerprint']==fingerprint(target)
            with (file.parent/'checkpoint.pkl').open('rb') as f:state=pickle.load(f)
            assert state['round']==50 and state['fingerprint']==r['fingerprint'];p=m.model_predict_scores(state['params'],tx);d=pd.read_csv(file.parent/'predictions.csv')
            np.testing.assert_array_equal(d.y_true,y);np.testing.assert_allclose(d.probability,p,atol=1e-10);np.testing.assert_array_equal(d.row_id,np.arange(len(y)));np.testing.assert_allclose(d.raw_LOC,loc)
            vals=[]
            for c in clients:
                v=pd.read_csv(file.parent/('validation_'+c['name']+'.csv'));vp=m.model_predict_scores(state['params'],c['VX']);np.testing.assert_array_equal(v.y_true,c['vy']);np.testing.assert_allclose(v.probability,vp,atol=1e-10);vals.append((c['vy'],vp))
            ts=dict(study_mcc=chosen_threshold(vals),fixed_05=.5,validation_f1=f1_threshold(vals))
            for k,v in effort(y,p,loc).items():assert r['effort'][k] is None if v is None else np.isclose(r['effort'][k],v)
            for mode,t in ts.items():
                assert np.isclose(t,r[mode]['threshold'])
                for k,v in classification(y,p,t).items():assert np.isclose(v,r[mode][k])
                rows.append(dict(target=target,family=task['family'],seed=r['seed'],condition=mode,**r[mode],**r['effort']))
            verified.append((target,r['seed']))
    if rows:
        df=pd.DataFrame(rows);df.to_csv(ROOT/'all_runs.csv',index=False);num=[n for n in df if n not in ['target','family','seed','condition']]
        df.groupby(['family','target','condition'])[num].agg(['mean','std','count']).to_csv(ROOT/'project_summary.csv')
        project=df.groupby(['family','target','condition'])[num].mean().reset_index();project.groupby(['family','condition'])[num].mean().to_csv(ROOT/'family_summary.csv')
    assert len(verified)==len(set(verified));save(ROOT/'verification.json',dict(verified_runs=len(verified),expected=300,status='complete' if len(verified)==300 else 'partial'));print('Verified',len(verified),'/300')

if __name__=='__main__':main()
