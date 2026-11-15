"""Independent numerical and frozen-data checks; no training/test-set tuning."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from prepare import keys,scale,content_hash
from train import weights,gradient
from metrics import classification,effort
ROOT=Path(__file__).resolve().parent

def main():
    # Hand-computed Eq 3-6 example: the globally minority label is 1.
    a,details=weights([np.r_[np.zeros(90,int),np.ones(10,int)],np.r_[np.zeros(50,int),np.ones(50,int)]])
    entropy=-(.9*np.log2(.9)+.1*np.log2(.1))
    expected=np.array([.5*entropy/6,.5*5/6]);expected/=expected.sum()
    np.testing.assert_allclose(a,expected);assert details['minority_label']==1
    np.testing.assert_allclose(weights([np.array([0,1])]*3)[0],np.ones(3)/3)
    # Finite-difference objective gradient, away from hinge kinks.
    x=np.array([[.2,.8],[.9,.1],[.3,.4]]);y=np.array([1,0,1]);w=np.array([.2,-.1]);b=.05;l2=.03
    def objective(v):return np.maximum(0,1-(2*y-1)*(x@v[:2]+v[2])).mean()+l2/2*np.dot(v[:2],v[:2])
    vec=np.r_[w,b];eps=1e-6
    numeric=np.array([(objective(vec+eps*np.eye(3)[i])-objective(vec-eps*np.eye(3)[i]))/(2*eps) for i in range(3)])
    gw,gb=gradient(w,b,x,y,l2);np.testing.assert_allclose(np.r_[gw,gb],numeric,atol=1e-8)
    assert classification(np.array([0,1]),np.array([.1,.9]),.5)['MCC']==1
    assert effort(np.array([0,1]),np.array([.1,.9]),np.array([0.,1.]))['Popt@20'] is None
    manifest=json.loads((ROOT/'data_manifest.json').read_text());tasks=json.loads((ROOT/'tasks.json').read_text())
    frames={};groups={};ratios=[]
    for n,r in manifest.items():
        path=ROOT/'data'/(n+'.csv');assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
        frames[n]=pd.read_csv(path);groups[n]=keys(frames[n])
    for target,task in tasks.items():
        a=np.load(ROOT/'prepared'/(target+'.npz'));assert content_hash(a)==task['prepared_content_sha256']
        splits=json.loads((ROOT/'prepared'/(target+'_splits.json')).read_text());trkeys=set();vakeys=set();cols=task['features']
        assert splits==json.loads((ROOT/'split_reference'/(target+'_splits.json')).read_text())
        assert set(task['sources'])=={n for n in frames if n!=target and manifest[n]['family']==task['family']}
        for i,n in enumerate(task['sources']):
            s=splits[n];tr=np.array(s['train']);va=np.array(s['validation']);excluded=s['excluded_target_feature_duplicates']
            assert not set(tr)&set(va);assert not (set(tr)|set(va))&set(excluded)
            assert set(tr)|set(va)|set(excluded)==set(range(len(frames[n])))
            assert set(excluded)==set(np.flatnonzero(np.isin(groups[n],groups[target])))
            assert not set(groups[n][tr])&set(groups[n][va])
            trkeys.update(groups[n][tr]);vakeys.update(groups[n][va]);ratios.append(len(va)/(len(tr)+len(va)))
            x=frames[n][cols].to_numpy(float);y=frames[n].defect_label.to_numpy(int)
            np.testing.assert_array_equal(a[f'y_{i}'],y[tr]);np.testing.assert_array_equal(a[f'v_{i}'],y[va])
            assert len(np.unique(y[tr]))==len(np.unique(y[va]))==2
            np.testing.assert_allclose(a[f'X_{i}'],scale(x[tr],x[tr]));np.testing.assert_allclose(a[f'V_{i}'],scale(x[tr],x[va]))
        # Across clients, feature overlap is explicitly permitted by this protocol.
        assert not (trkeys|vakeys)&set(groups[target])
        x=frames[target][cols].to_numpy(float);np.testing.assert_allclose(a['X_test'],scale(x,x))
        np.testing.assert_array_equal(a['y_test'],frames[target].defect_label.to_numpy(int))
        np.testing.assert_array_equal(a['loc'],frames[target][manifest[target]['loc']].to_numpy(float))
    print(json.dumps(dict(status='passed',targets=len(tasks),client_splits=len(ratios),validation_fraction_min=min(ratios),validation_fraction_max=max(ratios),global_feature_isolation_claimed=False,checks=['aggregation formula','hinge gradient','metrics sanity','raw hashes','all row assignments','within-client group isolation','source-target exclusion','exact frozen FLR indices','normalization reconstruction','prepared hashes'])))

if __name__=='__main__':main()
