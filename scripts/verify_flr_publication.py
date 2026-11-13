"""Data-free checks for the separately published completed FLR experiment."""
import hashlib
import json
import math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments/homogeneous/flr-tff-20261111'
OUT=ROOT/'results/flr-homogeneous-20261111'
meta=json.loads((EXP/'publication_provenance.json').read_text(encoding='utf-8'))
for name,expected in meta['published_file_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
tasks=json.loads((EXP/'tasks.json').read_text(encoding='utf-8'))
records=json.loads((OUT/'metrics_records.json').read_text(encoding='utf-8'))
expected={(t,s) for t in tasks for s in range(10)}
assert len(records)==300 and {(r['target'],r['seed']) for r in records}==expected
assert all(r['rounds']==100 and not r['pilot'] for r in records)
summary=json.loads((OUT/'analysis.json').read_text(encoding='utf-8'))
for mode in ['study','native']:
    for metric,value in summary['overall_project_macro'][mode].items():
        means=[]
        for t in tasks:
            vs=[r[mode].get(metric,r['effort'].get(metric)) for r in records if r['target']==t]
            vs=[v for v in vs if v is not None]
            if vs:means.append(sum(vs)/len(vs))
        assert math.isclose(sum(means)/len(means),value,abs_tol=1e-12),metric
assert summary['effort_projects']==13
assert len(summary['cross_client_train_validation_overlap_features'])==21
print('PASS: FLR published hashes, 300 distinct runs, 100 rounds, macro means and audit disclosures')
