"""Exercise the author's TFF API and an actual federated training round."""
import collections
import json
import platform
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

tf.random.set_seed(0)
options = tf.data.Options()
options.threading.private_threadpool_size = 1
data = []
raw_clients = [([[0., 0.], [1., 1.]], [[0.], [1.]]),
               ([[0., 1.], [1., 0.], [1., 1.]], [[0.], [1.], [1.]])]
for xs, ys in raw_clients:
    ds = tf.data.Dataset.from_tensor_slices(collections.OrderedDict(
        x=np.array(xs, dtype=np.float32), y=np.array(ys, dtype=np.float32)))
    data.append(ds.repeat(2).batch(2).with_options(options))

def model_fn():
    model = tf.keras.Sequential([tf.keras.layers.Dense(
        1, activation=tf.nn.sigmoid, kernel_initializer='zeros', input_shape=(2,))])
    return tff.learning.from_keras_model(
        model, input_spec=data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy()])

process = tff.learning.build_federated_averaging_process(
    model_fn=model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(1.0),
    use_experimental_simulation_loop=True)
state = process.initialize()
state, metrics = process.next(state, data)
weights = [np.asarray(a).tolist() for a in state.model.trainable]
assert all(np.isfinite(np.asarray(a)).all() for a in state.model.trainable)
assert any(np.any(np.asarray(a) != 0) for a in state.model.trainable)
# Independent numerical check of minibatch SGD and sample-weighted aggregation.
local = []
for xs, ys in raw_clients:
    x = np.tile(np.array(xs, dtype=np.float32), (2, 1))
    y = np.tile(np.array(ys, dtype=np.float32), (2, 1))
    w = np.zeros((2, 1), dtype=np.float32)
    b = np.zeros(1, dtype=np.float32)
    for start in range(0, len(y), 2):
        xb, yb = x[start:start+2], y[start:start+2]
        error = 1/(1+np.exp(-(xb@w+b))) - yb
        w -= .1 * xb.T@error/len(yb)
        b -= .1 * error.mean(axis=0)
    local.append((w, b, len(y)))
total = sum(v[2] for v in local)
for index in [0, 1]:
    expected = sum(v[index]*v[2]/total for v in local)
    np.testing.assert_allclose(np.asarray(state.model.trainable[index]), expected, atol=1e-6, rtol=1e-5)
print(json.dumps(dict(status='TFF_federated_round_passed', python=platform.python_version(),
    tensorflow=tf.__version__, tensorflow_federated=tff.__version__, weights=weights,
    metrics=str(metrics)), indent=2))
