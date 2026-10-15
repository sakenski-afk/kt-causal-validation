"""
RQ2 strengthened: pooled (three rolling-origin folds) implied vs causal lag curves, overall and by previous outcome.

    python rq2_folds_curves.py <rq1_dir> <rq2_dir>

Causal: fold estimates (spec B for 'all', spec C for after-correct / after-error) pooled with weights = test
re-exposures per fold and bin; variance = sum_f w_f^2 Var_f (folds treated as independent; see
rq2_bootstrap_folds.py). Models: implied effects summed over the folds' test re-exposures (GBM, DAS3H
row level; DKT-time 40k sample per fold). Writes rq2_folds_curves.csv (model, group, lag_bin, effect, ci_low, ci_high, n)
and rq2_folds_consistency_by_group.csv (re-exposure-weighted RMSE per group).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_models import LAG_BINS, lag_bin  # noqa: E402

LAGS = [lab for _, _, lab in LAG_BINS]
FOLDS = [("2025-07", "fold_2025-07"), ("2025-10", "fold_2025-10"), ("", "")]
GROUPS = {"all": ("B lag", lambda l: "lag " + l, None),
          "after correct": ("C lag x prev", lambda l: "after correct, lag " + l, 1),
          "after error": ("C lag x prev", lambda l: "after error, lag " + l, 0)}


def main(rq1, rq2):
    rows = []
    acc = {}  # (source, group, bin) -> [sum, n] for models; causal -> list of (mu, var, n)
    for tag, sub in FOLDS:
        d = os.path.join(rq2, sub) if sub else rq2
        sfx = "_test" + ("_" + tag if tag else "")
        ce = pd.read_csv(os.path.join(rq1, f"rq1_effects{sfx}.csv"))
        ce = ce[ce.estimator.str.startswith("E1")].set_index(["spec", "term"])
        s1 = pd.read_parquet(os.path.join(d, "rq2_test_predictions.parquet"))
        s1["lag_bin"] = lag_bin(np.where(s1.exposure > 1, np.expm1(s1.log_gap_h), -1))
        s1["imp_M5"] = s1["M5 GBM (all features)"] - s1["M5 GBM (all features) | cf unseen"]
        d3 = pd.read_parquet(os.path.join(d, "rq2_rows_das3h.parquet"))
        dk = pd.read_parquet(os.path.join(d, "rq2_implied_rows_deep_DKT-time.parquet"))
        for g, (spec, term, pv) in GROUPS.items():
            seen = s1[(s1.exposure > 1) & ((s1.prev_y == pv) if pv is not None else True)]
            d3s = d3[(d3.exposure > 1) & ((d3.prev_y == pv) if pv is not None else True)]
            dks = dk[(dk.prev_y == pv)] if pv is not None else dk
            for l in LAGS:
                n = int((seen.lag_bin == l).sum())
                r = ce.loc[(spec, term(l))]
                acc.setdefault(("causal", g, l), []).append((r.coef, r.se ** 2, n))
                for src, frame, col in (("M5 GBM (all features)", seen, "imp_M5"),
                                        ("M6b DAS3H + item windows", d3s, "implied | M6b DAS3H + item windows"),
                                        ("M7 DKT-time", dks, "implied")):
                    v = frame.loc[frame.lag_bin == l, col]
                    a = acc.setdefault((src, g, l), [0.0, 0])
                    a[0] += v.sum(); a[1] += len(v)
    for (src, g, l), v in acc.items():
        if src == "causal":
            n = np.array([x[2] for x in v], float); w = n / n.sum()
            mu = float(np.sum(w * np.array([x[0] for x in v])))
            se = float(np.sqrt(np.sum(w ** 2 * np.array([x[1] for x in v]))))
            rows.append(dict(model="CAUSAL", group=g, lag_bin=l, effect=mu, ci_low=mu - 1.96 * se, ci_high=mu + 1.96 * se, n=int(n.sum())))
        else:
            rows.append(dict(model=src, group=g, lag_bin=l, effect=v[0] / v[1] if v[1] else np.nan, ci_low=np.nan, ci_high=np.nan, n=v[1]))
    cur = pd.DataFrame(rows)
    cur.to_csv(os.path.join(rq2, "rq2_folds_curves.csv"), index=False, float_format="%.5f")
    cons = []
    for g in GROUPS:
        c = cur[(cur.model == "CAUSAL") & (cur.group == g)].set_index("lag_bin").loc[LAGS]
        for m in ("M5 GBM (all features)", "M6b DAS3H + item windows", "M7 DKT-time"):
            mm = cur[(cur.model == m) & (cur.group == g)].set_index("lag_bin").loc[LAGS]
            cons.append(dict(group=g, model=m, rmse_pp=100 * np.sqrt(np.average((mm.effect - c.effect) ** 2, weights=c.n)),
                             share_of_reexposures=c.n.sum()))
    cons = pd.DataFrame(cons)
    tot = cons[cons.group == "all"].share_of_reexposures.iloc[0]
    cons["share_of_reexposures"] = cons.share_of_reexposures / tot
    cons.to_csv(os.path.join(rq2, "rq2_folds_consistency_by_group.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(cons.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    p = cur.pivot_table(index=["group", "lag_bin"], columns="model", values="effect") * 100
    print(p.round(1).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
