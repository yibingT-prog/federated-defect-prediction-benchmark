"""Verify published hashes, frozen matrix, predictions, thresholds and summaries."""
import hashlib,json,sys,math,statistics as st
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/homogeneous/fedcpdp-native-20261114'
OUT=ROOT/'results/fedcpdp-native-homogeneous-20261114'
sys.path.insert(0,str(EXP))
from metrics import classification,effort,chosen_threshold
from train import f1_threshold,fingerprint

def main():
    for n,h in json.loads((OUT/'publication_manifest.json').read_text())['files'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    records=json.loads((OUT/'metrics_records.json').read_text());tasks=json.loads((EXP/'tasks.json').read_text())
    assert len(records)==300 and {(r['target'],r['seed']) for r in records}=={(t,s) for t in tasks for s in range(10)}
    for r in records:
        assert r['rounds']==50 and r['fingerprint']==fingerprint(r['target'])
        d=OUT/'runs'/r['target']/str(r['seed']);p=pd.read_csv(d/'predictions.csv')
        assert np.isfinite(p.probability).all() and p.probability.between(0,1).all()
        vals=[]
        splits=json.loads((EXP/'splits'/(r['target']+'_splits.json')).read_text())
        for source in tasks[r['target']]['sources']:
            v=pd.read_csv(d/('validation_'+source+'.csv'));np.testing.assert_array_equal(v.row_id,splits[source]['validation']);vals.append((v.y_true.to_numpy(),v.probability.to_numpy()))
        thresholds=dict(study_mcc=chosen_threshold(vals),fixed_05=.5,validation_f1=f1_threshold(vals))
        for mode,t in thresholds.items():
            assert np.isclose(t,r[mode]['threshold'])
            for k,v in classification(p.y_true.to_numpy(),p.probability.to_numpy(),t).items():assert np.isclose(v,r[mode][k]),(r['target'],mode,k)
        for k,v in effort(p.y_true.to_numpy(),p.probability.to_numpy(),p.raw_LOC.to_numpy()).items():
            assert r['effort'][k] is None if v is None else np.isclose(v,r['effort'][k])
    a=json.loads((OUT/'analysis.json').read_text())
    for mode,ms in a['overall_project_macro'].items():
        for m,v in ms.items():assert math.isclose(v,st.mean(st.mean(r[mode][m] for r in records if r['target']==t) for t in tasks),abs_tol=1e-12)
    assert len(pd.read_csv(OUT/'all_runs.csv'))==900
    print('PASS: publication hashes, 300 unique models, frozen fingerprints, predictions/thresholds/metrics, project macro means')
if __name__=='__main__':main()
