"""Native TFF FLR; source-only selection; target evaluation after final round."""
import argparse
import collections
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, confusion_matrix, matthews_corrcoef, f1_score
import tensorflow as tf
import tensorflow_federated as tff

ROOT = Path(__file__).resolve().parent

def save(path, obj):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)

def classify(y, p, threshold):
    pred = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(ROC_AUC=float(roc_auc_score(y, p)), MCC=float(matthews_corrcoef(y, pred)),
                Recall=float(tp/(tp+fn)), G_mean=float(np.sqrt(tp/(tp+fn)*tn/(tn+fp))),
                F1=float(f1_score(y, pred, zero_division=0)), threshold=float(threshold))

def effort(y, p, loc):
    keys = [k+'@'+str(b) for k in ['CostEffect', 'Popt'] for b in [5,10,20,40]]
    if not np.isfinite(loc).all() or np.any(loc <= 0) or y.sum() == 0:
        return dict(reason='Nonpositive/missing LOC or no positive labels', **dict.fromkeys(keys))
    def curve(order, b):
        x = np.r_[0, np.cumsum(loc[order])/loc.sum()]
        z = np.r_[0, np.cumsum(y[order])/y.sum()]
        xb = np.r_[x[x < b], b]
        zb = np.interp(xb, x, z)
        return float(zb[-1]), float(np.trapz(zb, xb))
    ids = np.arange(len(y)); out = {}
    for b in [5,10,20,40]:
        ce, area = curve(np.lexsort((ids, -p/loc)), b/100)
        _, best = curve(np.lexsort((ids, -y/loc)), b/100)
        _, worst = curve(np.lexsort((ids, y/loc)), b/100)
        out['CostEffect@'+str(b)] = ce
        out['Popt@'+str(b)] = (area-worst)/(best-worst) if abs(best-worst)>1e-12 else None
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rounds', type=int, default=100)
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    out = ROOT / ('pilot' if args.pilot else 'runs') / args.target / str(args.seed)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists():
        print('Already complete', args.target, args.seed, flush=True); return
    protocol = json.loads((ROOT/'protocol.json').read_text())
    cfg = protocol['training']
    task = json.loads((ROOT/'tasks.json').read_text())[args.target]
    arrays = np.load(ROOT/'prepared'/(args.target+'.npz'))
    train = [(arrays['X_'+str(i)], arrays['y_'+str(i)]) for i in range(len(task['sources']))]
    dim = train[0][0].shape[1]
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    def datasets(round_num):
        result = []
        for i, (x, y) in enumerate(train):
            ds = tf.data.Dataset.from_tensor_slices(collections.OrderedDict(
                x=x, y=y.astype(np.float32).reshape(-1, 1)))
            ds = ds.repeat(cfg['local_epochs']).shuffle(cfg['shuffle_buffer'],
                seed=args.seed*100000+round_num*100+i, reshuffle_each_iteration=True)
            result.append(ds.batch(cfg['batch_size']).with_options(options))
        return result
    first = datasets(1)
    def model_fn():
        keras = tf.keras.Sequential([tf.keras.layers.Dense(1, activation=tf.nn.sigmoid,
            kernel_initializer='zeros', input_shape=(dim,))])
        return tff.learning.from_keras_model(keras, loss=tf.keras.losses.BinaryCrossentropy(),
            input_spec=first[0].element_spec, metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])
    process = tff.learning.build_federated_averaging_process(model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(cfg['client_lr']),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(cfg['server_lr']),
        use_experimental_simulation_loop=True)
    state = process.initialize()
    started = time.time()
    save(out/'progress.json', dict(status='running', round=0, target=args.target, seed=args.seed))
    for r in range(1, args.rounds+1):
        state, met = process.next(state, first if r == 1 else datasets(r))
        if r == 1 or r % 5 == 0 or r == args.rounds:
            weights = [np.asarray(w) for w in state.model.trainable]
            assert all(np.isfinite(w).all() for w in weights)
            np.savez(out/'latest_weights.npz', *weights)
            save(out/'progress.json', dict(status='running', round=r, target=args.target,
                seed=args.seed, operational_seconds=time.time()-started, train_metrics=str(met)))
            print(args.target, args.seed, r, str(met), flush=True)
    weights = [np.asarray(w) for w in state.model.trainable]
    w, b = weights
    predict = lambda x: expit(x @ w + b).reshape(-1)
    vals = [(arrays['v_'+str(i)], predict(arrays['V_'+str(i)])) for i in range(len(train))]
    candidates = []
    for threshold in np.linspace(.05,.95,37):
        ms = [classify(y,p,threshold) for y,p in vals]
        candidates.append(((np.mean([m['MCC'] for m in ms]), np.mean([m['Recall'] for m in ms]), -threshold), threshold))
    threshold = float(max(candidates)[1])
    # The target is first scored after all predeclared training updates.
    y = arrays['y_test']; p = predict(arrays['X_test'])
    assert np.isfinite(p).all()
    pd.DataFrame(dict(row_id=np.arange(len(y)), y_true=y, score=p,
        prediction=(p>=threshold).astype(int))).to_csv(out/'predictions.csv',index=False)
    for i,(vy,vp) in enumerate(vals):
        pd.DataFrame(dict(y_true=vy,score=vp)).to_csv(out/('validation_'+task['sources'][i]+'.csv'),index=False)
    save(out/'metrics.json', dict(target=args.target,family=task['family'],seed=args.seed,
        rounds=args.rounds,pilot=args.pilot,clients=len(train),n_test=len(y),
        study=classify(y,p,threshold),native=classify(y,p,.5),effort=effort(y,p,arrays['loc']),
        tensorflow=tf.__version__,tensorflow_federated=tff.__version__))
    save(out/'progress.json',dict(status='completed',round=args.rounds,target=args.target,seed=args.seed))

if __name__ == '__main__':
    main()
