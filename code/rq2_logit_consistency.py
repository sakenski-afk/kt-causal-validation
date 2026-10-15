"""
RQ2 on the log-odds scale: model-implied effects vs conditional-logit causal effects (three test windows).

    python rq2_logit_consistency.py <rq1_dir> <rq2_dir> [n_boot] [--scale logit|pp] [--weights count|equal]

--scale pp: probability scale instead (reference = linear probability model E1, specs B/C per test window,
  rq1_effects_test*.csv with rq1_cov_test*_{B,C}.npy; implied effect = p - p_cf).
--weights equal: every lag bin with data gets the same weight in the RMSE (default: test re-exposures per bin),
  so that short intervals, where review decisions are made, count as much as long ones.
Outputs for non-default options: rq2_consistency_<scale>_<weights>_{summary,pairs,curves}.csv.

Why: on the probability scale the causal reference depends on the estimand (linear probability model vs
conditional logit differ by a factor 1.5-1.8 in percentage points). On the log-odds scale both sides are
conditional log-odds differences: the conditional logit within learner-games (rq1d_clogit.py --window)
gives the causal log-odds effect of a previous encounter per lag bin; for every model the implied effect is
logit(p_actual) - logit(p_counterfactual) on the same test re-exposures (probabilities clipped to [1e-4, 1-1e-4]).
Groups: all re-exposures (spec B), after a correct and after a wrong previous answer (spec C).
Models: M5 GBM, M6b DAS3H + item windows, M7 DKT-time (40k sample per window, p_act/p_cf saved by
rq2_deep.py), and models without an item path (implied 0). Bootstrap: learners resampled once across the
three windows (500 replicates); causal curves redrawn from each window's clogit covariance.
Outputs: rq2_logit_summary.csv, rq2_logit_pairs.csv, rq2_logit_curves.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.special import logit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_models import LAG_BINS, lag_bin  # noqa: E402

SEED = 42
SCALE, WEIGHTS = "logit", "count"
LAGS = [lab for _, _, lab in LAG_BINS]
MODELS = ["M5 GBM (all features)", "M6b DAS3H + item windows", "M7 DKT-time", "no item path (M5a, PFA, Elo, DKT, SAKT)"]
GROUPS = {"all": ("B", lambda l: "lag " + l, None),
          "after correct": ("C", lambda l: "after correct, lag " + l, 1),
          "after error": ("C", lambda l: "after error, lag " + l, 0)}
FOLDS = [("F1 2025-07..09", "fold_2025-07", "_test_2025-07"), ("F2 2025-10..12", "fold_2025-10", "_test_2025-10"),
         ("F3 2026-01..03", "", "_test")]


def lg(p):
    return logit(np.clip(p, 1e-4, 1 - 1e-4))


def main(rq1, rq2, n_boot=500):
    rng = np.random.default_rng(SEED)
    folds = []
    all_uids = []
    for name, sub, sfx in FOLDS:
        d = os.path.join(rq2, sub) if sub else rq2
        s1 = pd.read_parquet(os.path.join(d, "rq2_test_predictions.parquet"))
        s1 = s1[s1.exposure > 1].copy()
        s1["lag_bin"] = lag_bin(np.expm1(s1.log_gap_h))
        if SCALE == "pp":
            s1["imp"] = s1["M5 GBM (all features)"] - s1["M5 GBM (all features) | cf unseen"]
        else:
            s1["imp"] = lg(s1["M5 GBM (all features)"]) - lg(s1["M5 GBM (all features) | cf unseen"])
        d3 = pd.read_parquet(os.path.join(d, "rq2_rows_das3h.parquet"))
        d3 = d3[d3.exposure > 1].copy()
        p3 = d3["p | M6b DAS3H + item windows"]
        d3["imp"] = d3["implied | M6b DAS3H + item windows"] if SCALE == "pp" else lg(p3) - lg(p3 - d3["implied | M6b DAS3H + item windows"])
        dk = pd.read_parquet(os.path.join(d, "rq2_implied_rows_deep_DKT-time.parquet"))
        dk["imp"] = (dk.p_act - dk.p_cf) if SCALE == "pp" else lg(dk.p_act) - lg(dk.p_cf)
        rows = {"M5 GBM (all features)": s1[["uid", "lag_bin", "prev_y", "imp"]],
                "M6b DAS3H + item windows": d3[["uid", "lag_bin", "prev_y", "imp"]],
                "M7 DKT-time": dk[["uid", "lag_bin", "prev_y", "imp"]]}
        ref_rows = s1[["uid", "lag_bin", "prev_y"]]
        if SCALE == "pp":
            cl = pd.read_csv(os.path.join(rq1, f"rq1_effects{sfx}.csv"))
            cl = cl[cl.estimator.str.startswith("E1")].set_index(["spec", "term"])
            col, cpre = "coef", "rq1_cov"
        else:
            cl = pd.read_csv(os.path.join(rq1, f"rq1d_clogit{sfx}.csv")).set_index(["spec", "term"])
            col, cpre = "beta", "rq1d_cov"
        causal = {}
        for g, (spec, term, pv) in GROUPS.items():
            sname = "B lag" if spec == "B" else "C lag x prev"
            terms = pd.read_csv(os.path.join(rq1, f"{cpre}{sfx}_{spec}_terms.csv")).iloc[:, 0].tolist()
            V = np.load(os.path.join(rq1, f"{cpre}{sfx}_{spec}.npy"))
            ix = [terms.index(term(l)) for l in LAGS]
            mu = np.array([cl.loc[(sname, term(l)), col] for l in LAGS])
            causal[g] = (mu, V[np.ix_(ix, ix)])
        folds.append(dict(name=name, rows=rows, ref=ref_rows, causal=causal))
        all_uids.append(ref_rows.uid.values)
    uids = pd.Index(np.unique(np.concatenate(all_uids)))
    U = len(uids)

    def sums(df, pv):
        sel = df if pv is None else df[df.prev_y == pv]
        u = uids.get_indexer(sel.uid)
        S, N = np.zeros((U, len(LAGS))), np.zeros((U, len(LAGS)))
        for j, l in enumerate(LAGS):
            m = (sel.lag_bin.values == l) & (u >= 0)
            S[:, j] = np.bincount(u[m], sel.imp.values[m] if "imp" in sel else None, minlength=U)
            N[:, j] = np.bincount(u[m], minlength=U)
        return S, N

    for f in folds:
        f["S"], f["N"], f["draws"] = {}, {}, {}
        for g, (_, _, pv) in GROUPS.items():
            for m, df in f["rows"].items():
                f["S"][(m, g)], f["N"][(m, g)] = sums(df, pv)
            ref = f["ref"].assign(imp=0.0)
            _, f["N"][("ref", g)] = sums(ref, pv)
            mu, V = f["causal"][g]
            f["draws"][g] = rng.multivariate_normal(mu, V, size=n_boot)

    def stats(w_u, b):
        rm, imp, curves = {}, {}, {}
        for g in GROUPS:
            for m in MODELS:
                sq = wsum = 0.0; s_tot = n_tot = 0.0
                cs, cn = np.zeros(len(LAGS)), np.zeros(len(LAGS))
                for f in folds:
                    causal = f["causal"][g][0] if b is None else f["draws"][g][b]
                    wts = w_u @ f["N"][("ref", g)]
                    if WEIGHTS == "equal":
                        wts = (wts > 0).astype(float)
                    if m in f["rows"]:
                        s_b, n_b = w_u @ f["S"][(m, g)], w_u @ f["N"][(m, g)]
                        curve = np.divide(s_b, n_b, out=np.zeros_like(s_b), where=n_b > 0)
                        s_tot += s_b.sum(); n_tot += n_b.sum(); cs += s_b; cn += n_b
                    else:
                        curve = np.zeros(len(LAGS))
                    sq += (wts * (curve - causal) ** 2).sum(); wsum += wts.sum()
                rm[(m, g)] = np.sqrt(sq / wsum)
                imp[(m, g)] = s_tot / n_tot if n_tot else 0.0
                curves[(m, g)] = np.divide(cs, cn, out=np.zeros_like(cs), where=cn > 0)
        return rm, imp, curves

    r0, i0, c0 = stats(np.ones(U), None)
    keys = [(m, g) for g in GROUPS for m in MODELS]
    R, I = np.zeros((n_boot, len(keys))), np.zeros((n_boot, len(keys)))
    for b in range(n_boot):
        w_u = rng.multinomial(U, np.full(U, 1 / U)).astype(float)
        r, i, _ = stats(w_u, b)
        R[b], I[b] = [r[k] for k in keys], [i[k] for k in keys]
        if (b + 1) % 100 == 0:
            print("replicate", b + 1, flush=True)
    pct = lambda x: np.percentile(x, [2.5, 97.5], axis=0)
    rows = []
    for k_i, (m, g) in enumerate(keys):
        rows.append(dict(group=g, model=m, implied_logodds=i0[(m, g)], implied_lo=pct(I)[0][k_i], implied_hi=pct(I)[1][k_i],
                         rmse_logodds=r0[(m, g)], rmse_lo=pct(R)[0][k_i], rmse_hi=pct(R)[1][k_i]))
    summ = pd.DataFrame(rows)
    # pooled causal per group (re-exposure weighted) for reference
    for g in GROUPS:
        num = sum(f["causal"][g][0] * f["N"][("ref", g)].sum(0) for f in folds)
        den = sum(f["N"][("ref", g)].sum(0) for f in folds)
        summ.loc[summ.group == g, "causal_logodds_pooled"] = float(num.sum() / den.sum())
    summ.to_csv(os.path.join(rq2, OUT + "summary.csv"), index=False, float_format="%.4f")
    pairs = []
    for g in GROUPS:
        for ref in ("M5 GBM (all features)", "M6b DAS3H + item windows"):
            kr = keys.index((ref, g))
            for m in MODELS:
                if m == ref:
                    continue
                d = R[:, keys.index((m, g))] - R[:, kr]
                pairs.append(dict(group=g, reference=ref, model=m, d_rmse=r0[(m, g)] - r0[(ref, g)],
                                  lo=pct(d)[0], hi=pct(d)[1], P_model_better=float((d < 0).mean())))
    pd.DataFrame(pairs).to_csv(os.path.join(rq2, OUT + "pairs.csv"), index=False, float_format="%.4f")
    cur = []
    for g in GROUPS:
        num = sum(f["causal"][g][0] * f["N"][("ref", g)].sum(0) for f in folds)
        den = sum(f["N"][("ref", g)].sum(0) for f in folds)
        for j, l in enumerate(LAGS):
            cur.append(dict(group=g, lag_bin=l, model="CAUSAL clogit", logodds=num[j] / den[j]))
            for m in MODELS[:3]:
                cur.append(dict(group=g, lag_bin=l, model=m, logodds=c0[(m, g)][j]))
    pd.DataFrame(cur).to_csv(os.path.join(rq2, OUT + "curves.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(summ.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(pairs).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(pd.DataFrame(cur).pivot_table(index=["group", "lag_bin"], columns="model", values="logodds").round(2).to_string())


if __name__ == "__main__":
    a = sys.argv
    if "--scale" in a:
        SCALE = a[a.index("--scale") + 1]
    if "--weights" in a:
        WEIGHTS = a[a.index("--weights") + 1]
    OUT = "rq2_logit_" if (SCALE, WEIGHTS) == ("logit", "count") else f"rq2_consistency_{SCALE}_{WEIGHTS}_"
    main(a[1], a[2], int(a[3]) if len(a) > 3 and a[3].isdigit() else 500)
