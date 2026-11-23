import json,statistics,csv,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon
R=Path(__file__).resolve().parents[4]/'results/fedcpdp-privacy-utility-20261120'
O=Path(__file__).resolve().parents[4]/'results/fedcpdp-privacy-utility-20261120/report';O.mkdir(parents=True,exist_ok=True)
cs=json.loads((R/'study.json').read_text())['conditions']; manifest=json.loads((R/'conditions/none/data_manifest.json').read_text());ts=sorted(manifest)
keys=['ROC_AUC','MCC','Recall','G_mean','F1','CostEffect@20','Popt@20']
out={'conditions':cs,'manifest':manifest,'projects':{},'overall':{},'family':{},'contrasts':[],'diagnostics':{}}
arrays={}
for c in cs:
    projects={};logs=0;sat=[];boundary=0
    for t in ts:
        rs=[json.loads((R/'conditions'/c/'runs'/t/str(s)/'metrics.json').read_text()) for s in range(10)]
        assert all(r['rounds']==50 and r['target']==t for r in rs)
        vals=[{**r['study_mcc'],**r['effort']} for r in rs]
        projects[t]={k:{'mean':float(np.mean([v[k] for v in vals])),'sd':float(np.std([v[k] for v in vals],ddof=1))} if vals[0][k] is not None else None for k in keys}
        for s,r in enumerate(rs):
            folder=R/'conditions'/c/'runs'/t/str(s)
            logs+=int('overflow encountered in exp' in (folder/'console.log').read_text(errors='replace'))
            boundary+=int(np.isclose(r['study_mcc']['threshold'],.05) or np.isclose(r['study_mcc']['threshold'],.95))
            with (folder/'predictions.csv').open(newline='') as f:p=[float(x['probability']) for x in csv.DictReader(f)]
            sat.append(sum(x<=1e-6 or x>=1-1e-6 for x in p)/len(p))
    out['projects'][c]=projects;out['diagnostics'][c]={'overflow_warning_runs':logs,'boundary_threshold_runs':boundary,'mean_extreme_probability_fraction':float(np.mean(sat))}
    out['overall'][c]={k:{'mean':float(np.mean([projects[t][k]['mean'] for t in ts if projects[t][k] is not None])),'mean_seed_sd':float(np.mean([projects[t][k]['sd'] for t in ts if projects[t][k] is not None])),'n_projects':sum(projects[t][k] is not None for t in ts)} for k in keys}
    out['family'][c]={f:{k:float(np.mean([projects[t][k]['mean'] for t in ts if manifest[t]['family']==f and projects[t][k] is not None])) for k in keys[:5]} for f in sorted(set(v['family'] for v in manifest.values()))}
    arrays[c]={k:np.array([projects[t][k]['mean'] for t in ts]) for k in keys[:5]}
rng=np.random.default_rng(20261123)
for c in cs[1:]:
    for k in keys[:4]:
        delta=arrays[c][k]-arrays['none'][k];boot=delta[rng.integers(0,30,(20000,30))].mean(1)
        p=float(wilcoxon(delta,method='approx').pvalue) if np.any(delta) else 1.
        out['contrasts'].append({'condition':c,'metric':k,'delta':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'p':p,'wins':int((delta>1e-12).sum()),'ties':int((abs(delta)<=1e-12).sum()),'losses':int((delta< -1e-12).sum())})
order=sorted(range(20),key=lambda i:out['contrasts'][i]['p']);prev=0
for rank,i in enumerate(order):
    prev=max(prev,min(1.,(20-rank)*out['contrasts'][i]['p']));out['contrasts'][i]['p_holm']=prev
out['data_total']=sum(v['n'] for v in manifest.values())
(O/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'overall':out['overall'],'diagnostics':out['diagnostics'],'contrasts':[v for v in out['contrasts'] if v['metric'] in ['ROC_AUC','MCC']]},indent=2))
