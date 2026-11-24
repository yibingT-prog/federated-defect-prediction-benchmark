import numpy as np
from privacy import condition, perturb_flat

out, diag = perturb_flat(np.array([3.0, 4.0]), condition("clip_only"), np.random.default_rng(0))
assert np.allclose(out, [0.6, 0.8]) and diag["post_clip_norm"] <= 1.0 + 1e-12
raw = np.array([0.2, -0.3])
same, _ = perturb_flat(raw, condition("none"), np.random.default_rng(1))
assert np.array_equal(raw, same)
try:
    perturb_flat(np.array([np.nan]), condition("clip_only"), np.random.default_rng(0))
except ValueError:
    pass
else:
    raise AssertionError("non-finite update accepted")
print("PASS: FLR privacy mechanism tests")
