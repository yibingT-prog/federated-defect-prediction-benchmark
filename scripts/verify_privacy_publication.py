"""Verify the complete privacy publication without raw data or pickle loading."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/privacy/fedcpdp-native-20261120'
OUT=ROOT/'results/fedcpdp-privacy-utility-20261120'
sys.path.insert(0,str(EXP/'conditions/none'))
from metrics import classification,effort,chosen_threshold
from train import f1_threshold
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    for n,h in read(OUT/'publication_manifest.json')['files'].items():
        assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    original=read(EXP/'package_manifest.json')
    for n,h in original.items():
        if '/data/' not in n:assert hashlib.sha256((EXP/n).read_bytes()).hexdigest()==h,n
    cs=read(EXP/'study.json')['conditions'];analysis=read(OUT/'report/analysis.json')
    expected={'none':(False,0.),'clip_only':(True,0.),'sigma_0p1':(True,.1),'sigma_0p25':(True,.25),'sigma_0p5':(True,.5),'sigma_1p0':(True,1.)}
    total=0
    for c in cs:
        exp=EXP/'conditions'/c;out=OUT/'conditions'/c;cfg=read(exp/'protocol.json');tasks=read(exp/'tasks.json')
        assert cfg['privacy_condition']==c and cfg['model']['ROUNDS']==50
        assert (cfg['model']['USE_DP'],cfg['model']['DP_NOISE_MULTIPLIER'])==expected[c]
        assert cfg['model']['DP_CLIP_NORM']==1.
        records=read(out/'metrics_records.json');assert len(records)==300
        assert {(r['target'],r['seed']) for r in records}=={(t,s) for t in tasks for s in range(10)}
        hashes={}
        for t in tasks:
            names=['train.py','data.py','metrics.py','protocol.json','tasks.json','data_manifest.json','vendor/fed_cpdp.py','splits/'+t+'_splits.json']
            hashes[t]=hashlib.sha256(b''.join((exp/n).read_bytes() for n in names)).hexdigest()
        for r in records:
            t=r['target'];folder=out/'runs'/t/str(r['seed'])
            assert r==read(folder/'metrics.json') and r['rounds']==50 and r['fingerprint']==hashes[t]
            # Preserve near-one values: default CSV parsing can change AUC ties.
            p=pd.read_csv(folder/'predictions.csv',float_precision='round_trip');assert np.isfinite(p.probability).all() and p.probability.between(0,1).all()
            np.testing.assert_array_equal(p.row_id,np.arange(len(p)))
            vals=[];split=read(exp/'splits'/f'{t}_splits.json')
            for source in tasks[t]['sources']:
                v=pd.read_csv(folder/f'validation_{source}.csv',float_precision='round_trip')
                np.testing.assert_array_equal(v.row_id,split[source]['validation'])
                vals.append((v.y_true.to_numpy(),v.probability.to_numpy()))
            thresholds=dict(study_mcc=chosen_threshold(vals),fixed_05=.5,validation_f1=f1_threshold(vals))
            for mode,threshold in thresholds.items():
                assert np.isclose(threshold,r[mode]['threshold'])
                for k,value in classification(p.y_true.to_numpy(),p.probability.to_numpy(),threshold).items():
                    assert np.isclose(value,r[mode][k]),(c,t,r['seed'],mode,k)
            for k,value in effort(p.y_true.to_numpy(),p.probability.to_numpy(),p.raw_LOC.to_numpy()).items():
                assert r['effort'][k] is None if value is None else np.isclose(value,r['effort'][k])
        for k,value in analysis['overall'][c].items():
            means=[]
            for t in tasks:
                vs=[(r['study_mcc'] if k in r['study_mcc'] else r['effort'])[k] for r in records if r['target']==t]
                if vs[0] is not None:means.append(np.mean(vs))
            assert len(means)==value['n_projects'] and np.isclose(np.mean(means),value['mean'])
        assert len(pd.read_csv(out/'all_runs.csv'))==900
        assert read(out/'verification.json')['verified_runs']==300
        total+=300;print('PASS:',c,'300 models, all thresholds/metrics, fingerprints and macro means',flush=True)
    combined=pd.read_csv(OUT/'combined_all_runs.csv')
    assert len(combined)==5400 and set(combined.privacy_condition)==set(cs)
    assert set(combined.experiment_type)=={'privacy_utility_tradeoff'}
    assert combined.groupby(['privacy_condition','target','seed']).size().eq(3).all()
    print('PASS:',total,'models; privacy-utility publication is separate from homogeneous comparisons')
if __name__=='__main__':main()
