"""
RQ2: causal consistency of model-implied effects vs predictive accuracy.

    python rq2_consistency.py <rq1_dir> <rq2_dir>

Inputs: rq1_effects_test.csv (causal reference on the test window, E1, specs A-C) and
rq2_implied_effects.csv + rq2_metrics_test.csv (and, if present, *_extra files from later stages with
the same columns). For every model:
  bias_all        implied - causal for the overall re-exposure effect (spec A)
  rmse_lag        weighted RMSE over the 9 lag bins (spec B), weights = test re-exposures per bin
  rmse_lag_prev   weighted RMSE over the 18 previous-outcome x lag cells (spec C)
  coverage_lag    share of lag bins where the implied effect lies inside the causal 95% CI
Units: probability (x100 = percentage points). Ranking by AUC vs ranking by rmse_lag is reported.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def main(rq1, rq2):
    ce = pd.read_csv(os.path.join(rq1, "rq1_effects_test.csv"))
    ce = ce[ce.estimator.str.startswith("E1") & ~ce.term.str.startswith("ctrl")]
    causal = {}
    for _, r in ce.iterrows():
        key = "all" if r.term == "seen" else r.term
        causal[key] = (r.coef, r.ci_low, r.ci_high)
    imp = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(rq2, "rq2_implied_effects*.csv")))])
    met = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(rq2, "rq2_metrics_test*.csv")))])
    rows, curves = [], []
    for model, g in imp.groupby("model", sort=False):
        g = g.set_index("subset")
        def cells(prefix_filter):
            ks = [k for k in g.index if prefix_filter(k) and k in causal and g.loc[k, "n"] > 0]
            w = g.loc[ks, "n"].values.astype(float)
            d = g.loc[ks, "implied_effect"].values - np.array([causal[k][0] for k in ks])
            inside = [causal[k][1] <= g.loc[k, "implied_effect"] <= causal[k][2] for k in ks]
            return ks, w, d, inside
        ks, w, d, inside = cells(lambda k: k.startswith("lag "))
        _, w2, d2, _ = cells(lambda k: k.startswith("after "))
        rows.append(dict(model=model, implied_all=g.loc["all", "implied_effect"], causal_all=causal["all"][0],
                         bias_all=g.loc["all", "implied_effect"] - causal["all"][0],
                         rmse_lag=np.sqrt(np.average(d ** 2, weights=w)),
                         rmse_lag_prev=np.sqrt(np.average(d2 ** 2, weights=w2)),
                         coverage_lag=np.mean(inside)))
        for k in ks:
            curves.append(dict(model=model, subset=k, implied=g.loc[k, "implied_effect"], causal=causal[k][0],
                               ci_low=causal[k][1], ci_high=causal[k][2], n=int(g.loc[k, "n"])))
    res = pd.DataFrame(rows).merge(met[["model", "AUC", "LogLoss"]], on="model", how="left")
    res["rank_AUC"] = res.AUC.rank(ascending=False, method="min").astype(int)
    res["rank_consistency"] = res.rmse_lag.rank(ascending=True, method="min").astype(int)
    rho = spearmanr(res.AUC, -res.rmse_lag).correlation
    res.to_csv(os.path.join(rq2, "rq2_consistency.csv"), index=False, float_format="%.5f")
    pd.DataFrame(curves).to_csv(os.path.join(rq2, "rq2_curves_vs_causal.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 220)
    show = res.copy()
    for c in ("implied_all", "causal_all", "bias_all", "rmse_lag", "rmse_lag_prev"):
        show[c] = (show[c] * 100).round(2)
    print("effects in percentage points")
    print(show.sort_values("AUC", ascending=False).to_string(index=False))
    print("Spearman(AUC, -rmse_lag) =", round(rho, 3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
