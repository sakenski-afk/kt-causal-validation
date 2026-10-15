"""
Calibration slope and intercept (logistic recalibration: logit P(y=1) = a + b * logit(p)) for every model on the
test window F3, with learner-clustered bootstrap 95% CIs. Ideal: a = 0, b = 1 (b < 1: over-confident predictions).

    python review_calibration_slope.py <rq2_dir> <out_dir> [n_boot]
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_metrics_bootstrap import load_window  # noqa: E402


def recal(y, p, w):
    z = np.log(p / (1 - p))[:, None]
    m = LogisticRegression(C=1e6, solver="lbfgs").fit(z, y, sample_weight=w)
    return m.intercept_[0], m.coef_[0, 0]


def main(rq2, out, n_boot=200):
    os.makedirs(out, exist_ok=True)
    base, models = load_window(rq2)
    y = base.y.values.astype(int)
    u = pd.factorize(base.uid)[0]; U = u.max() + 1
    rng = np.random.default_rng(42)
    W = [rng.multinomial(U, np.full(U, 1 / U)).astype(float)[u] for _ in range(n_boot)]
    rows = []
    for m in models:
        p = np.clip(base[m].values, 1e-6, 1 - 1e-6)
        a, b = recal(y, p, np.ones(len(y)))
        bs = np.array([recal(y, p, w) for w in W])
        rows.append(dict(model=m, intercept=a, intercept_lo=np.percentile(bs[:, 0], 2.5), intercept_hi=np.percentile(bs[:, 0], 97.5),
                         slope=b, slope_lo=np.percentile(bs[:, 1], 2.5), slope_hi=np.percentile(bs[:, 1], 97.5)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "calibration_slope_intercept_F3.csv"), index=False, float_format="%.4f")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
