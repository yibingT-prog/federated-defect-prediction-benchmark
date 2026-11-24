"""Native TFF FLR privacy extension with client-side update perturbation."""
import argparse, collections, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, f1_score
import tensorflow as tf
import tensorflow_federated as tff

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1] / "homogeneous" / "flr-tff-20261111"
sys.path.insert(0, str(HERE))
from privacy import condition, ClientUpdateAggregationFactory


def save(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def classify(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(ROC_AUC=float(roc_auc_score(y, p)), MCC=float(matthews_corrcoef(y, pred)),
                Recall=float(tp/(tp+fn)), G_mean=float(np.sqrt(tp/(tp+fn)*tn/(tn+fp))),
                F1=float(f1_score(y, pred, zero_division=0)), threshold=float(threshold))


def data_hash(target):
    return hashlib.sha256((BASE / 'prepared' / (target + '.npz')).read_bytes()).hexdigest()


def fingerprint(cond, task, target):
    payload = b''.join((HERE / x).read_bytes() for x in ('train.py', 'privacy.py', 'protocol.json'))
    payload += (BASE / 'train_tff.py').read_bytes()
    payload += json.dumps(task, sort_keys=True).encode()
    payload += data_hash(target).encode()
    return hashlib.sha256(payload + cond.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', required=True); ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--condition', required=True); ap.add_argument('--rounds', type=int, default=100)
    ap.add_argument('--pilot', action='store_true')
    a = ap.parse_args()
    if not 0 <= a.seed <= 9: raise ValueError('seed must be 0..9')
    cfg = json.loads((HERE / 'protocol.json').read_text())
    task = json.loads((BASE / 'tasks.json').read_text())[a.target]
    cond = condition(a.condition)
    out = HERE / ('pilot' if a.pilot else 'runs') / a.condition / a.target / str(a.seed)
    out.mkdir(parents=True, exist_ok=True)
    prepared_hash = data_hash(a.target)
    fp = fingerprint(a.condition, task, a.target)
    if (out / 'metrics.json').exists():
        old = json.loads((out / 'metrics.json').read_text())
        if old.get('fingerprint') == fp and old.get('rounds') == a.rounds:
            print('SKIP completed', flush=True); return
    arrays = np.load(BASE / 'prepared' / (a.target + '.npz'))
    options = tf.data.Options(); options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    train = [(arrays['X_' + str(i)], arrays['y_' + str(i)]) for i in range(len(task['sources']))]
    dim = train[0][0].shape[1]
    # Match the native baseline's framework RNG initialization. Dataset
    # shuffle seeds remain derived per round/client below.
    tf.random.set_seed(a.seed)
    np.random.seed(a.seed)
    def datasets(round_num):
        result = []
        for i, (x, y) in enumerate(train):
            ds = tf.data.Dataset.from_tensor_slices(collections.OrderedDict(
                x=x, y=y.astype(np.float32).reshape(-1, 1)))
            ds = ds.repeat(cfg['training']['local_epochs']).shuffle(cfg['training']['shuffle_buffer'],
                seed=a.seed * 100000 + round_num * 100 + i, reshuffle_each_iteration=True)
            result.append(ds.batch(cfg['training']['batch_size']).with_options(options))
        return result
    first = datasets(1)
    def model_fn():
        keras = tf.keras.Sequential([tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
            kernel_initializer='zeros', input_shape=(dim,))])
        return tff.learning.from_keras_model(keras, loss=tf.keras.losses.BinaryCrossentropy(),
            input_spec=first[0].element_spec, metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    factory = None if cond.name == 'none' else ClientUpdateAggregationFactory(tff, cond, len(train))
    process = tff.learning.build_federated_averaging_process(model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(cfg['training']['client_lr']),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(cfg['training']['server_lr']),
        model_update_aggregation_factory=factory, use_experimental_simulation_loop=True)
    state = process.initialize(); started = time.time()
    save(out / 'progress.json', dict(status='running', round=0, target=a.target, seed=a.seed))
    # ``--pilot`` only selects the isolated pilot output tree.  The handoff
    # requires pilot targets to use the complete 100-round budget; the old
    # two-round smoke behavior is covered by the separate launch checks.
    rounds = a.rounds
    for r in range(1, rounds + 1):
        state, met = process.next(state, first if r == 1 else datasets(r))
        if r == 1 or r % 5 == 0 or r == rounds:
            weights = [np.asarray(w) for w in state.model.trainable]
            if not all(np.isfinite(w).all() for w in weights): raise FloatingPointError('non-finite model')
            np.savez(out / 'latest_weights.npz', *weights)
            save(out / 'progress.json', dict(status='running', round=r, target=a.target, seed=a.seed,
                operational_seconds=time.time()-started, train_metrics=str(met)))
            print(a.condition, a.target, a.seed, r, flush=True)
    if a.pilot:
        save(out / 'progress.json', dict(status='pilot_passed', round=rounds, target=a.target, seed=a.seed))
        return
    weights = [np.asarray(w) for w in state.model.trainable]; w, b = weights
    predict = lambda x: expit(x @ w + b).reshape(-1)
    vals = [(arrays['v_' + str(i)], predict(arrays['V_' + str(i)])) for i in range(len(train))]
    candidates = []
    for threshold in np.linspace(.05, .95, 37):
        ms = [classify(y, p, threshold) for y, p in vals]
        candidates.append(((np.mean([m['MCC'] for m in ms]), np.mean([m['Recall'] for m in ms]), -threshold), threshold))
    threshold = float(max(candidates)[1]); y = arrays['y_test']; p = predict(arrays['X_test'])
    pd.DataFrame(dict(row_id=np.arange(len(y)), y_true=y, score=p, prediction=(p >= threshold).astype(int))).to_csv(out / 'predictions.csv', index=False)
    for i, source in enumerate(task['sources']):
        pd.DataFrame(dict(y_true=vals[i][0], score=vals[i][1])).to_csv(out / ('validation_' + source + '.csv'), index=False)
    save(out / 'metrics.json', dict(experiment_type=cfg['experiment_type'], method='FLR', privacy_condition=a.condition,
        target=a.target, family=task['family'], seed=a.seed, rounds=rounds, fingerprint=fp,
        prepared_sha256=prepared_hash,
        study=classify(y, p, threshold), native=classify(y, p, .5), clients=len(train), n_test=len(y),
        tensorflow=tf.__version__, tensorflow_federated=tff.__version__,
        auxiliary_statistics_protected=False, formal_epsilon_delta_verified=False))
    save(out / 'progress.json', dict(status='completed', round=rounds, target=a.target, seed=a.seed))


if __name__ == '__main__': main()
