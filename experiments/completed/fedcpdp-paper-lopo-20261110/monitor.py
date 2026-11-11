"""Monitor this finite run and verify finished predictions; no scheduling service."""
from pathlib import Path
import json,time,hashlib,pickle
import numpy as np
import pandas as pd
from run import classification,effort_metrics,write_json
ROOT=Path(__file__).resolve().parent

def validate():
    cfg=json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    fingerprint=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in ['run.py','protocol.json','manifest.json','vendor/fed_cpdp.py'])).hexdigest()
    projects={}
    for name,meta in manifest.items():
        raw=pd.read_csv(ROOT/'data'/f'{name}.csv')
        assert hashlib.sha256((ROOT/'data'/f'{name}.csv').read_bytes()).hexdigest()==meta['sha256']
        scores=[]; records=[]
        for seed in cfg['seeds']:
            d=ROOT/'runs'/name/str(seed)
            pred=pd.read_csv(d/'predictions.csv'); rec=json.loads((d/'metrics.json').read_text())
            with (d/'checkpoint.pkl').open('rb') as f: ck=pickle.load(f)
            assert ck['round']==50 and ck['fingerprint']==fingerprint,(name,seed,'checkpoint')
            assert len(pred)==len(raw) and np.array_equal(pred.row_id,np.arange(len(raw)))
            assert np.array_equal(pred.y_true,raw.defect_label)
            assert np.allclose(pred.raw_LOC,raw[meta['loc']],equal_nan=True)
            assert np.isfinite(pred.score).all() and pred.score.between(0,1).all()
            for mode in ['study_mcc','paper_f1']:
                recomputed=classification(pred.y_true.to_numpy(),pred.score.to_numpy(),rec[mode]['threshold'])
                for k,v in recomputed.items():
                    assert v is None and rec[mode][k] is None or v is not None and np.isclose(v,rec[mode][k],atol=1e-10),(name,seed,mode,k)
            recomputed=effort_metrics(pred.y_true.to_numpy(),pred.score.to_numpy(),pred.raw_LOC.to_numpy())
            for k,v in recomputed.items():
                assert (v==rec['effort'][k]) if isinstance(v,str) or v is None else np.isclose(v,rec['effort'][k],atol=1e-10),(name,seed,k)
            scores.append(pred.score.to_numpy()); records.append(rec)
        ep=pd.read_csv(ROOT/'runs'/name/'ensemble_predictions.csv')
        assert np.array_equal(ep.y_true,raw.defect_label) and np.array_equal(ep.row_id,np.arange(len(raw)))
        assert np.allclose(ep.score,np.mean(scores,axis=0),atol=1e-12)
        er=json.loads((ROOT/'runs'/name/'ensemble_metrics.json').read_text())
        for mode in ['study_mcc','paper_f1']:
            recomputed=classification(ep.y_true.to_numpy(),ep.score.to_numpy(),er[mode]['threshold'])
            for k,v in recomputed.items(): assert np.isclose(v,er[mode][k],atol=1e-10),(name,'ensemble',mode,k)
        projects[name]={'family':meta['family'],'n':meta['n'],'invalid_LOC':meta['invalid_loc'],'seeds':records,'ensemble':er}
    summary={'status':'completed_and_verified','completed_unix':time.time(),'projects_count':len(projects),'base_models':len(projects)*len(cfg['seeds']),'project_details':projects,'protocol':'protocol.json','notes':cfg['deviations_and_limits']}
    write_json(ROOT/'verified_results.json',summary)
    return {'status':'completed_and_verified','projects':len(projects),'models':len(projects)*len(cfg['seeds']),'effort_eligible_projects':sum(m['invalid_loc']==0 for m in manifest.values())}

last=None; last_stale=False
while True:
    s=json.loads((ROOT/'status.json').read_text()); age=time.time()-s['updated_unix']
    state=(tuple(s['completed']),tuple(s['failed']),s['status'])
    if state!=last:
        print(json.dumps({'status':s['status'],'completed_projects':len(s['completed']),'failed':s['failed'],'active':list(s['active']),'queued':len(s['queued']),'models':len(list((ROOT/'runs').glob('*/*/metrics.json')))},ensure_ascii=False),flush=True); last=state
    stale=age>300
    if stale and not last_stale: print(json.dumps({'alert':'stale_scheduler','age_seconds':age}),flush=True)
    last_stale=stale
    if s['failed']:
        print('ATTENTION: failures require review',flush=True)
        break
    if s['status']=='completed':
        try: print(json.dumps(validate()),flush=True)
        except Exception:
            import traceback
            write_json(ROOT/'verification_failure.json',{'traceback':traceback.format_exc()})
            traceback.print_exc()
        break
    time.sleep(20)
