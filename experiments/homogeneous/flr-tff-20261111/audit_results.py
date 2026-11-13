"""Audit trusted local completed outputs and emit aggregate-only publication data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, f1_score

def classification(y, p, t):
    tn,fp,fn,tp=confusion_matrix(y,p>=t,labels=[0,1]).ravel()
    return dict(ROC_AUC=float(roc_auc_score(y,p)),MCC=float(matthews_corrcoef(y,p>=t)),Recall=float(tp/(tp+fn)),G_mean=float(np.sqrt(tp/(tp+fn)*tn/(tn+fp))),F1=float(f1_score(y,p>=t,zero_division=0)))

def effort(y,p,loc):
    if not np.isfinite(loc).all() or (loc<=0).any() or y.sum()==0:
        return {f'{m}@{b}':None for m in ['CostEffect','Popt'] for b in [5,10,20,40]}
    ids=np.arange(len(y)); result={}
    def area(order,b):
        x=np.r_[0,np.cumsum(loc[order])/loc.sum()];z=np.r_[0,np.cumsum(y[order])/y.sum()]
        xx=np.r_[x[x<b],b];zz=np.interp(xx,x,z)
        return float(zz[-1]),float(np.trapezoid(zz,xx))
    for b in [5,10,20,40]:
        ce,a=area(np.lexsort((ids,-p/loc)),b/100)
        _,best=area(np.lexsort((ids,-y/loc)),b/100);_,worst=area(np.lexsort((ids,y/loc)),b/100)
        result[f'CostEffect@{b}']=ce;result[f'Popt@{b}']=(a-worst)/(best-worst) if abs(best-worst)>1e-12 else None
    return result

def chosen_threshold(vals):
    choices=[]
    for t in np.linspace(.05,.95,37):
        mm=[];rr=[]
        for y,p in vals:
            z=p>=t;tp=int(np.sum(z&(y==1)));fp=int(np.sum(z&(y==0)));fn=int(np.sum(~z&(y==1)));tn=int(np.sum(~z&(y==0)))
            den=np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
            mm.append((tp*tn-fp*fn)/den if den else 0.);rr.append(tp/(tp+fn))
        choices.append(((np.mean(mm),np.mean(rr),-t),float(t)))
    return max(choices)[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,required=True);ap.add_argument('--raw-data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    root=args.run_root;tasks=json.loads((root/'tasks.json').read_text());manifest=json.loads((root/'data_manifest.json').read_text());records=[];file_hashes={};leakage={}
    keys={}
    for n,rec in manifest.items():
        p=args.raw_data/(n+'.csv');assert hashlib.sha256(p.read_bytes()).hexdigest()==rec['sha256']
        d=pd.read_csv(p);keys[n]=pd.util.hash_pandas_object(d.drop(columns='defect_label').reindex(sorted(c for c in d if c!='defect_label'),axis=1),index=False).astype(str).to_numpy()
    for target,task in tasks.items():
        a=np.load(root/'prepared'/(target+'.npz'));split=json.loads((root/'prepared'/(target+'_splits.json')).read_text())
        train=set();val=set()
        for n in task['sources']:
            tr=set(keys[n][split[n]['train']]);va=set(keys[n][split[n]['validation']]);assert not tr&va;assert not (tr|va)&set(keys[target]);train|=tr;val|=va
        if train&val:leakage[target]=len(train&val)
        for seed in range(10):
            folder=root/'runs'/target/str(seed);r=json.loads((folder/'metrics.json').read_text());assert r['target']==target and r['seed']==seed and r['rounds']==100 and not r['pilot']
            progress=json.loads((folder/'progress.json').read_text());assert progress['status']=='completed' and progress['round']==100
            df=pd.read_csv(folder/'predictions.csv');y=df.y_true.to_numpy();p=df.score.to_numpy();np.testing.assert_array_equal(y,a['y_test']);np.testing.assert_array_equal(df.row_id,np.arange(len(y)))
            w=np.load(folder/'latest_weights.npz');np.testing.assert_allclose(p,expit(a['X_test']@w['arr_0']+w['arr_1']).reshape(-1),atol=1e-6)
            vals=[]
            for i,n in enumerate(task['sources']):
                v=pd.read_csv(folder/('validation_'+n+'.csv'));np.testing.assert_array_equal(v.y_true,a['v_'+str(i)])
                np.testing.assert_allclose(v.score,expit(a['V_'+str(i)]@w['arr_0']+w['arr_1']).reshape(-1),atol=1e-6)
                vals.append((v.y_true.to_numpy(),v.score.to_numpy()))
            assert np.isclose(chosen_threshold(vals),r['study']['threshold'])
            assert r['native']['threshold']==.5
            for mode in ['study','native']:
                for k,v in classification(y,p,r[mode]['threshold']).items():np.testing.assert_allclose(v,r[mode][k],atol=1e-10)
            for k,v in effort(y,p,a['loc']).items():
                if v is None:assert r['effort'][k] is None
                else:np.testing.assert_allclose(v,r['effort'][k],atol=1e-10)
            records.append(r)
            for f in folder.glob('*'):
                if f.suffix in ['.json','.csv','.npz']:file_hashes[f.relative_to(root).as_posix()]=hashlib.sha256(f.read_bytes()).hexdigest()
        print('Verified',target,flush=True)
    assert len(records)==300 and len(list((root/'runs').glob('*/*/metrics.json')))==300
    rows=[dict(target=r['target'],family=r['family'],seed=r['seed'],condition=mode,**r[mode],**{k:v for k,v in r['effort'].items() if k!='reason'}) for r in records for mode in ['study','native']]
    df=pd.DataFrame(rows);metrics=['ROC_AUC','MCC','Recall','G_mean','F1','CostEffect@20','Popt@20']
    project=[]
    for (family,target,mode),g in df.groupby(['family','target','condition']):
        project.append(dict(family=family,target=target,condition=mode,mean={k:float(g[k].mean()) if g[k].notna().any() else None for k in metrics},sd={k:float(g[k].std(ddof=1)) if g[k].notna().any() else None for k in metrics}))
    families=[]
    for (family,mode),g in df.groupby(['family','condition']):
        pg=g.groupby('target')[metrics].mean();families.append(dict(family=family,condition=mode,projects=len(pg),effort_projects=int(pg['Popt@20'].notna().sum()),mean={k:float(pg[k].mean()) if pg[k].notna().any() else None for k in metrics}))
    overall={}
    for mode,g in df.groupby('condition'):
        pg=g.groupby('target')[metrics].mean();overall[mode]={k:float(pg[k].mean()) for k in metrics}
    summary=dict(runs=300,projects=30,seeds=list(range(10)),rounds=100,overall_project_macro=overall,families=families,projects_detail=project,effort_projects=sum(r['effort']['Popt@20'] is not None for r in records)//10,cross_client_train_validation_overlap_features=leakage,interpretation='Descriptive fixed-protocol results; inherited cross-client duplicate leakage is disclosed, not repaired retroactively. No FedCPDP ranking because protocols differ.')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,obj in [('analysis.json',summary),('metrics_records.json',records),('audit.json',dict(status='passed_numerical_and_coverage_checks',runs=300,checks=['all 100 rounds','predictions from final weights','validation predictions and labels','source-selected threshold','all classification metrics','all effort budgets'],cross_client_train_validation_overlap_features=leakage,local_output_sha256=file_hashes))]:
        (args.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(overall=overall,overlap=leakage),indent=2))

if __name__=='__main__':main()
