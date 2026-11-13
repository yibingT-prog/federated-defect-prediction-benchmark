"""Freeze same-family FLR tasks without changing prior experiments or raw data."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / 'fedcpdp-local-20261110'

def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')

def minmax(train, other):
    low = train.min(axis=0)
    span = train.max(axis=0) - low
    span[span == 0] = 1
    return np.asarray((other - low) / span, dtype=np.float32)

def main():
    manifest = json.loads((PRIOR / 'manifest.json').read_text(encoding='utf-8'))
    frames = {}
    for name, info in manifest.items():
        path = PRIOR / 'data' / (name + '.csv')
        assert hashlib.sha256(path.read_bytes()).hexdigest() == info['sha256']
        frames[name] = pd.read_csv(path)
        assert set(frames[name].defect_label.unique()) == {0, 1}
        assert np.isfinite(frames[name].to_numpy(dtype=float)).all()
    (ROOT / 'prepared').mkdir(exist_ok=True)
    tasks = {}
    for target, frame in frames.items():
        family = manifest[target]['family']
        sources = sorted(n for n in frames if n != target and manifest[n]['family'] == family)
        assert len(sources) >= 2
        cols = [c for c in frame.columns if c != 'defect_label']
        assert all(set(frames[n].columns) == set(frame.columns) for n in sources)
        prior_splits = json.loads((PRIOR / 'runs' / target / '42' / 'splits.json').read_text())
        splits = {name: prior_splits[name] for name in sources}
        xt = frame[cols].to_numpy(float)
        arrays = dict(X_test=minmax(xt, xt), y_test=frame.defect_label.to_numpy(int),
                      loc=frame[manifest[target]['loc']].to_numpy(float))
        for i, name in enumerate(sources):
            tr, va = splits[name]['train'], splits[name]['validation']
            assert not set(tr) & set(va)
            x = frames[name][cols].to_numpy(float)
            y = frames[name].defect_label.to_numpy(int)
            assert len(np.unique(y[tr])) == len(np.unique(y[va])) == 2
            arrays['X_' + str(i)] = minmax(x[tr], x[tr])
            arrays['V_' + str(i)] = minmax(x[tr], x[va])
            arrays['y_' + str(i)] = y[tr]
            arrays['v_' + str(i)] = y[va]
        np.savez_compressed(ROOT / 'prepared' / (target + '.npz'), **arrays)
        tasks[target] = dict(family=family, sources=sources, features=cols, n_test=len(frame),
            n_train=sum(len(splits[n]['train']) for n in sources),
            n_validation=sum(len(splits[n]['validation']) for n in sources))
        save(ROOT / 'prepared' / (target + '_splits.json'), splits)
    save(ROOT / 'tasks.json', tasks)
    save(ROOT / 'data_manifest.json', manifest)
    save(ROOT / 'protocol.json', dict(
        status='frozen_pending_TFF_smoke_test', model='FLR, native TensorFlow Federated API',
        data='Author FedCPDP CSV release, same-family project-LOPO; no hashing',
        training=dict(rounds=100, local_epochs=100, batch_size=20, shuffle_buffer=500,
                      client_lr=0.1, server_lr=1.0, participation=1.0, seeds=list(range(10)),
                      classifier='Dense(1, sigmoid), zero initial weights, binary crossentropy, SGD'),
        split='Reuse same-family source indices from prior FedCPDP run seed42; grouped ~80/20 split, seed20261110; target all records',
        normalization='Per-source minmax fitted only on training rows; source validation uses training scale; target uses its own unlabeled features',
        primary_threshold='Source-project macro MCC, then macro Recall, then lower threshold; 0.05..0.95 step0.025',
        native_threshold=0.5, checkpoint='Final actual aggregation round100; never test-selected',
        deviations=[
            'Author README TF2.5.0 conflicts with TFF0.20.0 metadata requiring TF2.8; use TF2.8.0 with TFF0.20.0',
            'Dataset release and project-level same-family protocol differ from original FLR benchmark',
            'Explicit held-out source validation and duplicate grouping/exclusion reused from prior run',
            '100 actual updates, correcting author range(1,100) which runs99',
            '10 seeded runs instead of original100; no ensemble',
            'Retain LOC-zero records for common test population; invalid LOC makes effort metrics unavailable',
            'Constant minmax columns handled with unit denominator instead of dropping rows',
            'Deterministic per-client/per-round shuffle seeds; repeat then shuffle then batch as author',
            'No LACE2 dependency or baselines in this FLR-only run'],
        metrics=['ROC_AUC', 'MCC', 'Recall', 'G_mean', 'F1', 'CostEffect@20', 'Popt@20'],
        efficiency_evaluation=False))
    print(json.dumps(dict(status='prepared', targets=len(tasks), runs=len(tasks)*10,
                         families=sorted(set(t['family'] for t in tasks.values())))))

if __name__ == '__main__':
    main()
