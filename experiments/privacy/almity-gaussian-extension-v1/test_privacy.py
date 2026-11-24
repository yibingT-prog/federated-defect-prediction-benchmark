import pathlib
import sys
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from privacy import condition, perturb_update


def main():
    rng = np.random.default_rng(7)
    w, b, d = perturb_update(np.array([3.0]), np.array(4.0), condition("clip_only"), rng)
    assert np.allclose([w[0], b], [0.6, 0.8])
    assert d["post_clip_norm"] <= 1.0 + 1e-12
    w, b, _ = perturb_update(np.array([0.2, -0.3]), np.array(0.4), condition("none"), rng)
    assert np.array_equal(w, [0.2, -0.3]) and b == 0.4
    try:
        perturb_update(np.array([np.nan]), np.array(0.0), condition("none"), rng)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite update accepted")
    print("PASS: ALMITY privacy mechanism tests")


if __name__ == "__main__":
    main()
