"""Verify published metrics, rows, histories and provenance without data or PyTorch."""
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def check(actual,expected):
    for k,v in expected.items():
        if v is None:assert actual[k] is None,k
        else:assert np.isclose(actual[k],v,rtol=1e-6,atol=1e-7),(k,actual[k],v)

def verify(method):
    exp=ROOT/'experiments/homogeneous'/f'{method}-resource-20261116';out=ROOT/'results'/f'{method}-homogeneous-20261116'
    manifest=read(out/'publication_manifest.json')
    for name,digest in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    cfg=read(exp/'protocol.json');tasks=read(exp/'tasks.json');data=read(exp/'data_manifest.json');target_splits=read(exp/'target_splits.json')
    provenance=read(exp/'provenance/source_snapshot.json')
    original=read(exp/'provenance/executed_package_manifest.json');package=read(exp/'package_manifest.json')
    assert package=={n:h for n,h in original.items() if not n.startswith('author_reference/')}
    for n,h in provenance['byte_identical_execution_files'].items():assert hashlib.sha256((exp/n).read_bytes()).hexdigest()==h
    spec=importlib.util.spec_from_file_location('resource_metrics',exp/'metrics.py');metric=importlib.util.module_from_spec(spec);spec.loader.exec_module(metric)
    records=read(out/'metrics_records.json');expected={(t,s) for t in tasks for s in cfg['seeds']}
    assert len(records)==cfg['expected_runs']==manifest['expected_runs']==len(expected)
    assert {(r['target'],r['seed']) for r in records}==expected
    all_runs=pd.read_csv(out/'all_runs.csv',float_precision='round_trip');assert len(all_runs)==len(expected)
    assert not all_runs.duplicated(['target','seed']).any()
    rows=[];local=read(out/'local_artifact_sha256.json')
    for r in records:
        t,s=r['target'],r['seed'];task=tasks[t];folder=out/'runs'/t/str(s)
        assert r==read(folder/'metrics.json') and r['rounds']==50 and r['family']==task['family']
        assert r['identity']==dict(fingerprint=r['identity']['fingerprint'],target=t,seed=s,rounds=50)
        assert r['identity']['fingerprint'] in provenance['executed_fingerprints']
        for p in folder.iterdir():assert hashlib.sha256(p.read_bytes()).hexdigest()==local[p.relative_to(out).as_posix()],p
        assert len(task['sources'])>=2 and t not in task['sources'] and task['public'] not in task['sources']
        p=pd.read_csv(folder/'predictions.csv',float_precision='round_trip');v=pd.read_csv(folder/'validation_predictions.csv',float_precision='round_trip')
        audit=read(folder/'data_audit.json');splits=read(exp/'splits'/f'{t}_splits.json');pairs=[]
        assert np.isfinite(p.probability).all() and p.probability.between(0,1).all()
        test_rows=target_splits[t]['query'] if method=='ecfml' else list(range(data[t]['n']))
        assert p.row.tolist()==test_rows==audit[t]['query'] and len(p)==r['test_n']
        assert set(v.project)==set(task['sources'])
        for n in task['sources']:
            z=v[v.project==n];assert z.row.tolist()==splits[n]['validation']==audit[n]['validation']
            assert audit[n]['train']==splits[n]['train'] and not set(audit[n]['train'])&set(audit[n]['validation'])
            assert np.isfinite(z.probability).all() and z.probability.between(0,1).all();pairs.append((z.y.to_numpy(),z.probability.to_numpy()))
        assert np.isclose(metric.chosen_threshold(pairs),r['threshold'])
        y=p.y.to_numpy();prob=p.probability.to_numpy();loc=p['loc'].to_numpy()
        assert r['effort_valid']==(data[t]['invalid_loc']==0)
        measured={**metric.classification(y,prob,r['threshold']),**metric.effort(y,prob,loc if r['effort_valid'] else np.full(len(y),np.nan))}
        check(r['metrics'],measured);check(r['native_threshold_metrics'],metric.classification(y,prob,.5))
        csvrow=all_runs[(all_runs.target==t)&(all_runs.seed==s)].iloc[0]
        for k,val in measured.items():assert pd.isna(csvrow[k]) if val is None else np.isclose(csvrow[k],val)
        for k,val in r['native_threshold_metrics'].items():assert np.isclose(csvrow['native_'+k],val)
        assert np.isclose(csvrow.threshold,r['threshold']) and csvrow.test_n==len(p)
        history=read(folder/'history.json');assert [h['round'] for h in history]==list(range(1,51))
        if method=='ecfml':
            support=target_splits[t]['support'];assert audit[t]['support']==support and len(support)==r['support_n']
            assert not set(support)&set(test_rows) and sorted(support+test_rows)==list(range(data[t]['n']))
            config=read(folder/'resolved_model_config.json');assert config['privacy']['enabled'] and config['privacy']['clipping_bound']==.0001
            assert config['training']['target_adaptation_steps']==1 and all(np.isfinite(h['meta_query_loss']) for h in history)
        else:
            assert audit['public']['labels_used'] is False and r['public_n']==data[task['public']]['n']
            counts=None
            for h in history:
                ids=h['public_indices'];assert len(ids)==len(set(ids))==min(700,r['public_n']) and min(ids)>=0 and max(ids)<r['public_n']
                assert set(h['aggregation_sample_counts'])==set(task['sources'])
                assert all(n>=len(splits[k]['train']) and n%2==0 for k,n in h['aggregation_sample_counts'].items())
                if counts is None:counts=h['aggregation_sample_counts']
                assert h['aggregation_sample_counts']==counts and np.isfinite(h['kd_loss'])
            last=read(folder/'last10_native_metrics.json');assert [h['round'] for h in last]==list(range(41,51))
            for entry in last:
                z=pd.read_csv(folder/f'predictions_round_{entry["round"]}.csv',float_precision='round_trip')
                assert z.row.tolist()==test_rows and np.array_equal(z.y,y);check(entry,metric.classification(y,z.probability.to_numpy(),.5))
            check(r['last10_native_mean'],{k:float(np.mean([e[k] for e in last])) for k in last[0] if k!='round'})
        rows.append(dict(family=r['family'],target=t,seed=s,**measured))
    df=pd.DataFrame(rows);cols=list(records[0]['metrics']);means=df.groupby(['family','target'])[cols].mean()
    reported=read(out/'verification.json')['overall_macro'];check(reported,{k:float(means[k].mean()) for k in reported})
    pm=pd.read_csv(out/'project_summary.csv').set_index(['family','target']);fm=pd.read_csv(out/'family_summary.csv').set_index('family')
    for k in ['ROC_AUC','MCC','Recall','G_mean','F1','CostEffect@20','Popt@20']:
        np.testing.assert_allclose(pm[k+'_mean'],means[k].reindex(pm.index),equal_nan=True)
        np.testing.assert_allclose(pm[k+'_std'],df.groupby(['family','target'])[k].std().reindex(pm.index),equal_nan=True)
        np.testing.assert_allclose(fm[k],means.groupby('family')[k].mean().reindex(fm.index),equal_nan=True)
    model_check=read(out/'model_verification.json');assert model_check['checked']==len(records) and model_check['all_models_match_prediction_files'] and not model_check['partial']
    print(f'PASS: {method}, {len(records)} runs, executed source bytes, predictions, thresholds, effort, row splits, 50-round histories and summaries')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--method',required=True,choices=['ecfml','feddp']);verify(ap.parse_args().method)
