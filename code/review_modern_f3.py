"""
MAKE revision: AKT and HawkesKT on the test window F3 (rows of Table 5 and of the accuracy/calibration table).

    python review_modern_f3.py <rq1_dir> <rq2_dir> <modern_F3_dir> [n_boot]

Common test rows as in review_calibration_slope.py (rq2_metrics_bootstrap.load_window); metrics with
baselines.metrics on all responses and on re-exposures; calibration slope with the same learner-clustered
bootstrap (seed 42); causal inconsistency on F3 = re-exposure-weighted RMSE of the implied lag curve (40k sample,
direct protocol) against the spec-B causal curve (rq1_effects_test.csv), and the number of lag bins inside its 95% CI.
Output: modern_f3.csv in <modern_F3_dir>.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402
from review_calibration_slope import recal  # noqa: E402
from rq2_bootstrap_folds import KEY, LAGS  # noqa: E402
from rq2_metrics_bootstrap import load_window  # noqa: E402
from rq2_models import lag_bin  # noqa: E402


def main(rq1, rq2, md, n_boot=200):
    base, _ = load_window(rq2)
    base = base.reset_index(drop=True)
    y = base.y.values.astype(int)
    u = pd.factorize(base.uid)[0]; U = u.max() + 1
    rng = np.random.default_rng(42)
    W = [rng.multinomial(U, np.full(U, 1 / U)).astype(float)[u] for _ in range(n_boot)]
    s1 = pd.read_parquet(os.path.join(rq2, "rq2_test_predictions.parquet"))
    s1 = s1[~s1.duplicated(KEY, keep=False)]
    s1["lag_bin"] = lag_bin(np.where(s1.exposure > 1, np.expm1(s1.log_gap_h), -1))
    nref = s1[s1.exposure > 1].lag_bin.value_counts().reindex(LAGS).values.astype(float)
    ce = pd.read_csv(os.path.join(rq1, "rq1_effects_test.csv"))
    ce = ce[(ce.spec == "B lag") & ce.estimator.str.startswith("E1")].set_index("term").loc[["lag " + l for l in LAGS]]
    rows = []
    for n in ("AKT", "HawkesKT"):
        p = pd.read_parquet(os.path.join(md, f"rq2_rows_deep_{n}.parquet"))
        p = p[~p.duplicated(KEY, keep=False)]
        pr = base[KEY].merge(p[KEY + ["p"]], on=KEY, how="left").p.values.astype(float)
        assert not np.isnan(pr).any()
        pr = np.clip(pr, 1e-6, 1 - 1e-6)
        seen = base.exposure.values > 1
        m_all, m_seen = B.metrics(y.astype(float), pr), B.metrics(y[seen].astype(float), pr[seen])
        a0, b0 = recal(y, pr, None)
        bs = np.array([recal(y, pr, w)[1] for w in W])
        imp = pd.read_parquet(os.path.join(md, f"rq2_implied_rows_deep_{n}.parquet"))
        curve = imp.groupby("lag_bin").implied.mean().reindex(LAGS).values
        D = np.sqrt(np.average((curve - ce.coef.values) ** 2, weights=nref))
        inside = int(((curve >= ce.coef.values - 1.96 * ce.se.values) & (curve <= ce.coef.values + 1.96 * ce.se.values)).sum())
        rows.append(dict(model=f"M7 {n}", n=len(y), AUC=m_all["AUC"], LogLoss=m_all["LogLoss"], RMSE=m_all["RMSE"],
                         ECE_all=m_all["ECE"], ECE_reexp=m_seen["ECE"], cal_intercept=a0, cal_slope=b0,
                         slope_lo=np.percentile(bs, 2.5), slope_hi=np.percentile(bs, 97.5),
                         implied_pp=100 * imp.implied.mean(), D_F3_pp=100 * D, bins_inside=f"{inside}/{len(LAGS)}"))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(md, "modern_f3.csv"), index=False, float_format="%.5f")


if __name__ == "__main__":
    main(*sys.argv[1:4], int(sys.argv[4]) if len(sys.argv) > 4 else 200)
