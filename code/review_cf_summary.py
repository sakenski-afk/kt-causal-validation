"""
Summary of the counterfactual-protocol check (test window F3): implied effect and causal inconsistency D_m per model
and protocol (direct item-memory vs full-history deletion), with the same reference and weights as the paper
(spec B, E1, F3 test responses; weights = test re-exposures per lag bin).

    python review_cf_summary.py <work_dir>
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_models import LAG_BINS, lag_bin  # noqa: E402

LAGS = [b for _, _, b in LAG_BINS]


def main(work):
    rq1, rq2, cf = (os.path.join(work, d) for d in ("rq1", "rq2", os.path.join("review", "cf")))
    eff = pd.read_csv(os.path.join(rq1, "rq1_effects_test.csv"))
    ref = eff[(eff.spec == "B lag") & eff.estimator.str.startswith("E1")].set_index("term").coef
    ref = pd.Series({b: ref[f"lag {b}"] for b in LAGS})
    pred = pd.read_parquet(os.path.join(rq2, "rq2_test_predictions.parquet"))
    rx = pred[pred.exposure > 1].copy()
    rx["lag_bin"] = lag_bin(np.expm1(rx.log_gap_h.values))
    w = rx.lag_bin.value_counts().reindex(LAGS).fillna(0).values

    curves = []

    def add(model, protocol, lag, implied):
        g = pd.DataFrame({"lag_bin": lag, "imp": implied}).groupby("lag_bin").imp.mean().reindex(LAGS)
        curves.append(dict(model=model, protocol=protocol, implied_all=float(np.average(g.fillna(0), weights=w)),
                           D_m_pp=100 * float(np.sqrt(np.average((g.fillna(0) - ref) ** 2, weights=w))),
                           **{f"imp {b}": 100 * g[b] for b in LAGS}))

    # PFA (no item-specific path): direct = 0 by construction, full = paper
    add("M3 PFA", "direct", rx.lag_bin, np.zeros(len(rx)))
    add("M3 PFA", "full", rx.lag_bin, (rx["M3 PFA"] - rx["M3 PFA | cf unseen"]).values)
    # GBM
    g = pd.read_parquet(os.path.join(cf, "cf_protocols_gbm_rows.parquet"))
    g["lag_bin"] = lag_bin(np.expm1(g.log_gap_h.values))
    add("M5 GBM", "direct", g.lag_bin, g["direct | pipeline"].values)
    add("M5 GBM", "full", g.lag_bin, g["full | pipeline"].values)
    add("M5 GBM (game-batched Elo)", "direct", g.lag_bin, g["direct | batched"].values)
    # DAS3H + item windows
    full = pd.read_parquet(os.path.join(rq2, "rq2_rows_das3h.parquet"))
    full = full[full.exposure > 1]
    add("M6b DAS3H + item windows", "full", full.lag_bin, full["implied | M6b DAS3H + item windows"].values)
    dpath = os.path.join(work, "review", "cf_das3h_direct", "rq2_rows_das3h.parquet")
    if os.path.exists(dpath):
        dd = pd.read_parquet(dpath)
        dd = dd[dd.exposure > 1]
        add("M6b DAS3H + item windows", "direct", dd.lag_bin, dd["implied | M6b DAS3H + item windows"].values)
    # deep models
    deep = pd.read_parquet(os.path.join(cf, "cf_protocols_deep_rows.parquet"))
    for (m, prot), d in deep.groupby(["model", "protocol"]):
        add(m, prot, d.lag_bin, d.implied.values)
    res = pd.DataFrame(curves)
    res.to_csv(os.path.join(cf, "cf_protocols_summary.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print("causal reference (pp):", (100 * ref).round(2).to_dict())
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1])
