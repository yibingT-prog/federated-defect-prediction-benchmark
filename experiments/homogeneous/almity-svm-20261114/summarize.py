"""Summarize completed jobs without treating seeds as independent projects."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from metrics import classification,effort,chosen_threshold
from train import fingerprint,save
ROOT=Path(__file__).resolve().parent
def main():
    tasks=json.loads((ROOT/'tasks.json').read_text());rows=[]
    for file in sorted((ROOT/'runs').glob('*/*/metrics.json')):
        r=json.loads(file.read_text());assert r['rounds']==100 and r['fingerprint']==fingerprint(tasks[r['target']]);d=pd.read_csv(file.parent/'predictions.csv');a=np.load(ROOT/'prepared'/(r['target']+'.npz'));w=np.load(file.parent/'checkpoint.npz');assert int(w['round'])==100
        np.testing.assert_array_equal(d.y_true,a['y_test']);np.testing.assert_allclose(d.margin,a['X_test']@w['w']+w['b'],atol=1e-10)
        np.testing.assert_allclose(d.probability,expit(r['calibration']['slope']*d.margin+r['calibration']['intercept']),atol=1e-10)
        vals=[pd.read_csv(file.parent/('validation_'+n+'.csv')) for n in tasks[r['target']]['sources']]
        for i,v in enumerate(vals):
            np.testing.assert_array_equal(v.y_true,a[f'v_{i}'])
            np.testing.assert_allclose(v.margin,a[f'V_{i}']@w['w']+w['b'],atol=1e-10)
            np.testing.assert_allclose(v.probability,expit(r['calibration']['slope']*v.margin+r['calibration']['intercept']),atol=1e-10)
        assert np.isclose(chosen_threshold([(v.y_true.to_numpy(),v.probability.to_numpy()) for v in vals]),r['study']['threshold'])
        for k,v in effort(d.y_true.to_numpy(),d.probability.to_numpy(),a['loc']).items():
            assert r['effort'][k] is None if v is None else np.isclose(r['effort'][k],v)
        for mode in ['study','fixed_probability','native_margin']:
            for k,v in classification(d.y_true.to_numpy(),d.margin.to_numpy() if mode=='native_margin' else d.probability.to_numpy(),r[mode]['threshold']).items():assert np.isclose(r[mode][k],v)
            rows.append(dict(target=r['target'],family=r['family'],seed=r['seed'],condition=mode,**r[mode],**r['effort']))
    if rows:
        df=pd.DataFrame(rows);df.to_csv(ROOT/'all_runs.csv',index=False);num=[c for c in df if c not in ['target','family','seed','condition']]
        df.groupby(['family','target','condition'])[num].agg(['mean','std','count']).to_csv(ROOT/'project_summary.csv')
        project_means=df.groupby(['family','target','condition'])[num].mean().reset_index()
        project_means.groupby(['family','condition'])[num].mean().to_csv(ROOT/'family_summary.csv')
    save(ROOT/'verification.json',dict(verified_runs=len(rows)//3,expected=300,status='complete' if len(rows)==900 else 'partial'));print('Verified',len(rows)//3,'/300')
if __name__=='__main__':main()
