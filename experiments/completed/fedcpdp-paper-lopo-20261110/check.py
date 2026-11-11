"""Preflight: all folds, leakage controls, and independently known metric cases."""
import json, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from run import ROOT,feature_keys,effort_metrics,classification,split_source
frames={p.stem:pd.read_csv(p) for p in (ROOT/'data').glob('*.csv')}
splits=[]
for target,tar in frames.items():
    keys=set(feature_keys(tar))
    for name,d in frames.items():
        if name==target: continue
        g=feature_keys(d); mask=~np.isin(g,list(keys)) if set(d.columns)==set(tar.columns) else np.ones(len(d),bool)
        y=d.defect_label.to_numpy()[mask]; g=g[mask]
        tr,va=split_source(y,g)
        assert len(np.unique(y[tr]))==len(np.unique(y[va]))==2,(target,name)
        assert not set(g[tr])&set(g[va])
        splits.append(dict(target=target,source=name,train=len(tr),validation=len(va),excluded=int((~mask).sum())))
y=np.array([1,0]); loc=np.array([1.,1.])
best=effort_metrics(y,np.array([.9,.1]),loc); worst=effort_metrics(y,np.array([.1,.9]),loc)
assert abs(best['CostEffect@20']-.4)<1e-12
assert abs(best['Popt@20']-1)<1e-12 and abs(worst['Popt@20'])<1e-12
assert effort_metrics(y,np.array([.9,.1]),np.array([0.,1.]))['Popt@20'] is None
assert classification(np.array([0,1]),np.array([.1,.1]),.5)['MCC']==0
(ROOT/'preflight.json').write_text(json.dumps({'status':'passed','source_target_pairs':len(splits),'metric_checks':'perfect/reverse order, fractional LOC budget, invalid LOC, constant classifier','splits':splits},indent=2))
print('PASS',len(splits),'source-target splits and metric edge cases')
