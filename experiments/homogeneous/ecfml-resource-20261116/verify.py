import argparse,hashlib,time
import numpy as np
import pandas as pd
from common import ROOT,read,save,load_data,fingerprint
from metrics import classification,effort,chosen_threshold

def compare(actual,expected):
    for k,v in expected.items():
        if v is None:assert actual[k] is None,k
        else:assert np.isclose(actual[k],v,rtol=1e-6,atol=1e-7),(k,actual[k],v)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');a=ap.parse_args()
    cfg=read(ROOT/'protocol.json');tasks=read(ROOT/'tasks.json'); audits={}
    for name,digest in read(ROOT/'package_manifest.json').items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    for t,task in tasks.items():
        _,_,c,d,p,audit=load_data(t)
        assert len(c)>=2
        assert all(n!=t and n!=task['public'] for n in c)
        if cfg['method']=='ecfml':
            assert not set(d['support'])&set(d['query'])
            assert len(d['support'])+len(d['query'])==read(ROOT/'data_manifest.json')[t]['n']
        audits[t]=dict(clients=len(c),query=len(d['y']),support=len(d['support']),public=len(p) if p is not None else 0,
                       public_overlap=audit.get('public'))
    if a.preflight:
        save(ROOT/'preflight.json',dict(ok=True,expected_runs=cfg['expected_runs'],tasks=audits,checked_unix=time.time()));print('preflight passed',len(tasks));return
    rows=[]; fp=fingerprint()
    for target,task in tasks.items():
        _,_,clients,td,_,_=load_data(target)
        for seed in cfg['seeds']:
            folder=ROOT/'runs'/target/str(seed); m=read(folder/'metrics.json')
            assert m['identity']==dict(fingerprint=fp,target=target,seed=seed,rounds=cfg['rounds'])
            history=read(folder/'history.json');assert [h['round'] for h in history]==list(range(1,cfg['rounds']+1))
            pred=pd.read_csv(folder/'predictions.csv');val=pd.read_csv(folder/'validation_predictions.csv')
            assert np.array_equal(pred.row,td['query']);assert np.array_equal(pred.y,td['y'])
            pairs=[]
            for n,c in clients.items():
                v=val[val.project==n];assert np.array_equal(v.row,c['validation']);assert np.array_equal(v.y,c['vy']);pairs.append((v.y.to_numpy(),v.probability.to_numpy()))
            assert len(val)==sum(len(c['vy']) for c in clients.values())
            threshold=chosen_threshold(pairs);assert np.isclose(threshold,m['threshold'])
            p=pred.probability.to_numpy();y=pred.y.to_numpy();loc=pred['loc'].to_numpy() if m['effort_valid'] else np.full(len(y),np.nan)
            compare(m['metrics'],{**classification(y,p,threshold),**effort(y,p,loc)})
            compare(m['native_threshold_metrics'],classification(y,p,.5))
            if cfg['method']=='feddp':
                supplemental=read(folder/'last10_native_metrics.json')
                assert [v['round'] for v in supplemental]==list(range(cfg['rounds']-9,cfg['rounds']+1))
                for entry in supplemental:
                    z=pd.read_csv(folder/f'predictions_round_{entry["round"]}.csv')
                    assert np.array_equal(z.row,td['query']) and np.array_equal(z.y,td['y'])
                    compare(entry,classification(z.y.to_numpy(),z.probability.to_numpy(),.5))
                compare(m['last10_native_mean'],{k:float(np.mean([v[k] for v in supplemental])) for k in supplemental[0] if k!='round'})
            rows.append(dict(target=target,family=task['family'],seed=seed,threshold=threshold,test_n=len(y),support_n=m['support_n'],**m['metrics'],**{'native_'+k:v for k,v in m['native_threshold_metrics'].items()}))
    out=pd.DataFrame(rows);assert len(out)==cfg['expected_runs'];out.to_csv(ROOT/'all_runs.csv',index=False)
    metrics=['ROC_AUC','MCC','Recall','G_mean','F1','CostEffect@20','Popt@20']
    projects=out.groupby(['family','target'])[metrics].agg(['mean','std']);projects.columns=['_'.join(x) for x in projects.columns];projects.to_csv(ROOT/'project_summary.csv')
    project_means=out.groupby(['family','target'])[metrics].mean();project_means.groupby('family').mean().to_csv(ROOT/'family_summary.csv')
    save(ROOT/'verification.json',dict(ok=True,runs=len(out),targets=len(tasks),overall_macro=project_means.mean().where(project_means.mean().notna(),None).to_dict(),checked_unix=time.time()))
    print('completed and independently verified',len(out),flush=True)

if __name__=='__main__':main()
