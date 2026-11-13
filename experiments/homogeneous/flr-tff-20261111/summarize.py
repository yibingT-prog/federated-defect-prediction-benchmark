"""Verify saved native-TFF predictions and summarize completed repetitions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, matthews_corrcoef, recall_score, f1_score

ROOT = Path(__file__).resolve().parent
def main():
    tasks = json.loads((ROOT/'tasks.json').read_text())
    rows = []; checked = []
    for path in sorted((ROOT/'runs').glob('*/*/metrics.json')):
        record = json.loads(path.read_text())
        assert record['rounds']==100 and not record['pilot']
        frame = pd.read_csv(path.parent/'predictions.csv')
        data = np.load(ROOT/'prepared'/(record['target']+'.npz'))
        y,p = frame.y_true.to_numpy(),frame.score.to_numpy()
        np.testing.assert_array_equal(y,data['y_test'])
        np.testing.assert_array_equal(frame.row_id.to_numpy(),np.arange(len(y)))
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        weights = np.load(path.parent/'latest_weights.npz')
        np.testing.assert_allclose(p,expit(data['X_test']@weights['arr_0']+weights['arr_1']).reshape(-1),atol=1e-6)
        for condition in ['study','native']:
            met = record[condition]; pred = p>=met['threshold']
            check = dict(ROC_AUC=roc_auc_score(y,p),MCC=matthews_corrcoef(y,pred),
                         Recall=recall_score(y,pred),F1=f1_score(y,pred,zero_division=0))
            for key,value in check.items(): np.testing.assert_allclose(met[key],value,atol=1e-10)
            rows.append(dict(target=record['target'],family=record['family'],seed=record['seed'],
                threshold_condition=condition,**met,**{k:v for k,v in record['effort'].items() if k!='reason'}))
        checked.append(dict(target=record['target'],seed=record['seed']))
    if rows:
        df = pd.DataFrame(rows); df.to_csv(ROOT/'all_runs.csv',index=False)
        metrics = [c for c in df if c not in ['target','family','seed','threshold_condition']]
        summary = df.groupby(['family','target','threshold_condition'])[metrics].agg(['mean','std','count'])
        summary.columns = ['_'.join(c) for c in summary.columns]
        summary.to_csv(ROOT/'project_summary.csv')
        df.groupby(['family','target','threshold_condition'])[metrics].mean().groupby(['family','threshold_condition']).mean().to_csv(ROOT/'family_summary.csv')
    (ROOT/'verification.json').write_text(json.dumps(dict(status='complete' if len(checked)==300 else 'partial',
        expected_runs=300,verified_runs=len(checked),verified_tasks=checked),indent=2),encoding='utf-8')
    print('Verified',len(checked),'of 300 runs',flush=True)

if __name__ == '__main__': main()
