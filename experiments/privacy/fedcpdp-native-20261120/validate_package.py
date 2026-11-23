"""Verify packaged bytes, shared data/splits, and native clipping/noise behavior."""
import hashlib,importlib.util,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def main():
    assert os.environ.get('PYTHONHASHSEED')=='0'
    for name,digest in json.loads((ROOT/'package_manifest.json').read_text()).items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    names=json.loads((ROOT/'study.json').read_text())['conditions'];base=ROOT/'conditions'/names[0]
    for name in names:
        folder=ROOT/'conditions'/name;cfg=json.loads((folder/'protocol.json').read_text())
        assert cfg['privacy_condition']==name and cfg['model']['ROUNDS']==50
        for sub in ['data','splits','vendor']:
            for p in (base/sub).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:assert p.read_bytes()==(folder/p.relative_to(base)).read_bytes()
        manifest=json.loads((folder/'data_manifest.json').read_text())
        for project,meta in manifest.items():assert hashlib.sha256((folder/'data'/f'{project}.csv').read_bytes()).hexdigest()==meta['sha256']
    spec=importlib.util.spec_from_file_location('native',base/'vendor/fed_cpdp.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    delta=(np.array([3.,4.]),); clipped=m.clip_and_add_noise(delta,1.,0.,np.random.RandomState(7))[0]
    np.testing.assert_allclose(clipped,[.6,.8]); np.testing.assert_array_equal(delta[0],[3.,4.])
    noisy=m.clip_and_add_noise(delta,1.,.5,np.random.RandomState(7))[0]
    expected=clipped+np.random.RandomState(7).normal(0.,.5,2);np.testing.assert_allclose(noisy,expected)
    print('PASS package hashes, frozen data/splits, native clip and Gaussian scale',flush=True)
if __name__=='__main__':main()
