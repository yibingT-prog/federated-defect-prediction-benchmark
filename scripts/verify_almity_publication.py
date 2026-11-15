"""Data-free verification of published ALMITY predictions and declared adapters."""
import json,hashlib,sys,statistics as st,math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
R=Path(__file__).resolve().parents[1];E=R/'experiments/homogeneous/almity-svm-20261114';O=R/'results/almity-homogeneous-20261114'
sys.path.insert(0,str(E))
from metrics import classification,effort,chosen_threshold
from train import fingerprint

def main():
    for n,h in json.loads((O/'publication_manifest.json').read_text())['files'].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h,n
    rs=json.loads((O/'metrics_records.json').read_text());tasks=json.loads((E/'tasks.json').read_text());manifest=json.loads((E/'data_manifest.json').read_text())
    assert len(rs)==300 and {(r['target'],r['seed']) for r in rs}=={(t,s) for t in tasks for s in range(10)}
    for r in rs:
        assert r['rounds']==100 and r['fingerprint']==fingerprint(tasks[r['target']])
        d=O/'runs'/r['target']/str(r['seed']);p=pd.read_csv(d/'predictions.csv');assert len(p)==manifest[r['target']]['n'];vals=[]
        for source in tasks[r['target']]['sources']:
            v=pd.read_csv(d/('validation_'+source+'.csv'));np.testing.assert_allclose(v.probability,expit(r['calibration']['slope']*v.margin+r['calibration']['intercept']),atol=1e-10);vals.append((v.y_true.to_numpy(),v.probability.to_numpy()))
        assert np.isfinite(p.probability).all();np.testing.assert_allclose(p.probability,expit(r['calibration']['slope']*p.margin+r['calibration']['intercept']),atol=1e-10)
        assert np.isclose(chosen_threshold(vals),r['study']['threshold'])
        for mode in ['study','fixed_probability','native_margin']:
            for k,v in classification(p.y_true.to_numpy(),p.margin.to_numpy() if mode=='native_margin' else p.probability.to_numpy(),r[mode]['threshold']).items():assert np.isclose(v,r[mode][k]),(r['target'],mode,k)
        for k,v in effort(p.y_true.to_numpy(),p.probability.to_numpy(),p.raw_LOC.to_numpy()).items():assert r['effort'][k] is None if v is None else np.isclose(v,r['effort'][k])
        agg=json.loads((d/'aggregation.json').read_text());b=np.asarray(agg['Phi'])*np.asarray(agg['alpha'])*np.asarray(agg['Psi']);np.testing.assert_allclose(b/b.sum(),agg['normalized'])
    a=json.loads((O/'analysis.json').read_text())
    for mode,ms in a['overall_project_macro'].items():
        for m,v in ms.items():assert math.isclose(v,st.mean(st.mean(r[mode][m] for r in rs if r['target']==t) for t in tasks),abs_tol=1e-12)
    assert len(pd.read_csv(O/'all_runs.csv'))==900
    print('PASS: ALMITY hashes, 300 distinct runs, 100 rounds, calibration, native metrics, thresholds, effort, aggregation and macro means')
if __name__=='__main__':main()
