"""Verify frozen source bytes and the published result coverage without data."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENT=ROOT/'experiments/completed/fedcpdp-paper-lopo-20261110'
p=json.loads((EXPERIMENT/'provenance.json').read_text(encoding='utf-8'))
for name,expected in p['source_files_sha256'].items():
    assert hashlib.sha256((EXPERIMENT/name).read_bytes()).hexdigest()==expected,name
results=json.loads((ROOT/'results/fedcpdp-paper-lopo-20261110/verified_results.json').read_text(encoding='utf-8'))
assert results['projects_count']==30 and results['base_models']==90
assert len(results['project_details'])==30
assert all(len(x['seeds'])==3 for x in results['project_details'].values())
assert results['ensemble_effort_verified'] is True
print('PASS: frozen source hashes and 30-project / 90-model result coverage')
