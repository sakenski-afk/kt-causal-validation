"""
RQ2 strengthened: pooled learner-clustered bootstrap over three rolling-origin test windows.

    python rq2_bootstrap_folds.py <rq1_dir> <rq2_dir> [n_boot]

Folds (train < T_VAL <= val < T_TEST <= test < END):
  F1  2025-04-01 / 2025-07-01 / 2025-10-01   -> <rq2>/fold_2025-07, causal rq1_effects_test_2025-07.csv
  F2  2025-07-01 / 2025-10-01 / 2026-01-01   -> <rq2>/fold_2025-10, causal rq1_effects_test_2025-10.csv
  F3  2025-10-01 / 2026-01-01 / 2026-04-01   -> <rq2>,              causal rq1_effects_test.csv (main split)
Models: M5 GBM, M5a GBM without memory, M6b DAS3H + item windows, M7 DKT-time (models without an item path
imply 0 and behave like M5a). Each replicate resamples learners once (multinomial weights shared across folds,
since learners appear in several windows) and redraws each fold's causal lag curve from N(beta, V).
Pooled metrics: AUC = test-size-weighted mean of fold AUCs; causal inconsistency = sqrt of the
re-exposure-weighted mean squared difference over fold x lag-bin cells. The folds' causal draws are taken as
independent (an approximation: overlapping learners make them mildly correlated).
Outputs: rq2_folds_summary.csv, rq2_folds_pairs.csv, rq2_folds_by_fold.csv (point estimates per fold).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_models import LAG_BINS, lag_bin  # noqa: E402

SEED = 42
KEY = ["uid", "game_id", "question_id"]
LAGS = [lab for _, _, lab in LAG_BINS]
MODELS = ["M5 GBM (all features)", "M5a GBM without memory features", "M6b DAS3H + item windows", "M7 DKT-time"]


def load_fold(rq1, rq2, tag):
    d = rq2 if tag == "" else os.path.join(rq2, "fold_" + tag)
    s1 = pd.read_parquet(os.path.join(d, "rq2_test_predictions.parquet"))
    s1 = s1[~s1.duplicated(KEY, keep=False)]
    s1["lag_bin"] = lag_bin(np.where(s1.exposure > 1, np.expm1(s1.log_gap_h), -1))
    d3 = pd.read_parquet(os.path.join(d, "rq2_rows_das3h.parquet"))
    d3 = d3[~d3.duplicated(KEY, keep=False)]
    dk = pd.read_parquet(os.path.join(d, "rq2_rows_deep_DKT-time.parquet")).rename(columns={"p": "p_dkt"})
    dk = dk[~dk.duplicated(KEY, keep=False)]
    base = s1.merge(d3[KEY + ["p | M6b DAS3H + item windows", "implied | M6b DAS3H + item windows"]], on=KEY) \
             .merge(dk[KEY + ["p_dkt"]], on=KEY)
    preds = {"M5 GBM (all features)": base["M5 GBM (all features)"].values,
             "M5a GBM without memory features": base["M5a GBM without memory features"].values,
             "M6b DAS3H + item windows": base["p | M6b DAS3H + item windows"].values,
             "M7 DKT-time": base["p_dkt"].values}
    implied = {"M5 GBM (all features)": (base["M5 GBM (all features)"] - base["M5 GBM (all features) | cf unseen"]).values,
               "M6b DAS3H + item windows": base["implied | M6b DAS3H + item windows"].values}
    ir = pd.read_parquet(os.path.join(d, "rq2_implied_rows_deep_DKT-time.parquet"))
    sfx = "_test" + ("" if tag == "" else "_" + tag)
    ce = pd.read_csv(os.path.join(rq1, f"rq1_effects{sfx}.csv"))
    ce = ce[(ce.spec == "B lag") & ce.estimator.str.startswith("E1")].set_index("term")
    terms = pd.read_csv(os.path.join(rq1, f"rq1_cov{sfx}_B_terms.csv")).iloc[:, 0].tolist()
    V = np.load(os.path.join(rq1, f"rq1_cov{sfx}_B.npy"))
    ix = [terms.index("lag " + l) for l in LAGS]
    return dict(base=base, preds=preds, implied=implied, dkt_rows=ir,
                mu=ce.loc[["lag " + l for l in LAGS], "coef"].values, V=V[np.ix_(ix, ix)])


def main(rq1, rq2, n_boot=500):
    rng = np.random.default_rng(SEED)
    folds = {"F1 2025-07..09": load_fold(rq1, rq2, "2025-07"), "F2 2025-10..12": load_fold(rq1, rq2, "2025-10"),
             "F3 2026-01..03": load_fold(rq1, rq2, "")}
    uids = pd.Index(np.unique(np.concatenate([f["base"].uid.values for f in folds.values()])))
    U = len(uids)
    for name, f in folds.items():
        b = f["base"]
        f["u"] = uids.get_indexer(b.uid)
        f["y"] = b.y.values
        seen = b.exposure.values > 1
        lb = b.lag_bin.values
        f["S"], f["N"] = {}, None
        Nref = np.zeros((U, len(LAGS)))
        for j, l in enumerate(LAGS):
            m = seen & (lb == l)
            Nref[:, j] = np.bincount(f["u"][m], minlength=U)
            for mod, imp in f["implied"].items():
                f["S"].setdefault(mod, np.zeros((U, len(LAGS))))[:, j] = np.bincount(f["u"][m], imp[m], minlength=U)
        f["N"] = Nref
        r = f["dkt_rows"]; ru = uids.get_indexer(r.uid); ok = ru >= 0
        Sd, Nd = np.zeros((U, len(LAGS))), np.zeros((U, len(LAGS)))
        for j, l in enumerate(LAGS):
            m = ok & (r.lag_bin.values == l)
            Sd[:, j] = np.bincount(ru[m], r.implied.values[m], minlength=U)
            Nd[:, j] = np.bincount(ru[m], minlength=U)
        f["S"]["M7 DKT-time"], f["N_dkt"] = Sd, Nd
        f["draws"] = rng.multivariate_normal(f["mu"], f["V"], size=n_boot)
        print(name, "rows", len(b), "re-exposures", int(seen.sum()), "learners", b.uid.nunique(), flush=True)

    def stats(w_u, b_idx):
        auc, sq, wsum, imp_tot, imp_n = {}, {m: 0.0 for m in MODELS}, 0.0, {m: 0.0 for m in MODELS}, {m: 0.0 for m in MODELS}
        auc = {m: [] for m in MODELS}; fsize = []
        for name, f in folds.items():
            w = w_u[f["u"]]; keep = w > 0
            fsize.append((w[keep]).sum())
            for m in MODELS:
                auc[m].append(roc_auc_score(f["y"][keep], f["preds"][m][keep], sample_weight=w[keep]))
            causal = f["mu"] if b_idx is None else f["draws"][b_idx]
            wts = w_u @ f["N"]
            wsum += wts.sum()
            for m in MODELS:
                if m in f["S"]:
                    N = f["N_dkt"] if m == "M7 DKT-time" else f["N"]
                    s_b, n_b = w_u @ f["S"][m], w_u @ N
                    curve = np.divide(s_b, n_b, out=np.zeros_like(s_b), where=n_b > 0)
                    imp_tot[m] += s_b.sum(); imp_n[m] += n_b.sum()
                else:
                    curve = np.zeros(len(LAGS))
                sq[m] += (wts * (curve - causal) ** 2).sum()
        fsize = np.array(fsize)
        return ({m: np.average(auc[m], weights=fsize) for m in MODELS},
                {m: np.sqrt(sq[m] / wsum) for m in MODELS},
                {m: imp_tot[m] / imp_n[m] if imp_n[m] else 0.0 for m in MODELS})

    a0, r0, i0 = stats(np.ones(U), None)
    A, R, I = np.zeros((n_boot, 4)), np.zeros((n_boot, 4)), np.zeros((n_boot, 4))
    for b in range(n_boot):
        w_u = rng.multinomial(U, np.full(U, 1 / U)).astype(float)
        a, r, i = stats(w_u, b)
        A[b], R[b], I[b] = [a[m] for m in MODELS], [r[m] for m in MODELS], [i[m] for m in MODELS]
        if (b + 1) % 50 == 0:
            print("replicate", b + 1, flush=True)
    pct = lambda x: np.percentile(x, [2.5, 97.5], axis=0)
    causal_all = sum(f["mu"] @ f["N"].sum(0) for f in folds.values()) / sum(f["N"].sum() for f in folds.values())
    summ = pd.DataFrame({"model": MODELS, "AUC": [a0[m] for m in MODELS], "AUC_lo": pct(A)[0], "AUC_hi": pct(A)[1],
                         "implied_all": [i0[m] for m in MODELS], "implied_lo": pct(I)[0], "implied_hi": pct(I)[1],
                         "rmse_lag": [r0[m] for m in MODELS], "rmse_lo": pct(R)[0], "rmse_hi": pct(R)[1],
                         "P(best AUC)": np.bincount(A.argmax(1), minlength=4) / n_boot,
                         "P(best consistency)": np.bincount(R.argmin(1), minlength=4) / n_boot})
    summ["causal_all_pooled"] = causal_all
    summ.to_csv(os.path.join(rq2, "rq2_folds_summary.csv"), index=False, float_format="%.5f")
    pairs = []
    for ref in ("M5 GBM (all features)", "M6b DAS3H + item windows"):
        k0 = MODELS.index(ref)
        for k, m in enumerate(MODELS):
            if k == k0:
                continue
            da, dr = A[:, k] - A[:, k0], R[:, k] - R[:, k0]
            pairs.append(dict(reference=ref, model=m, dAUC=a0[m] - a0[ref], dAUC_lo=pct(da)[0], dAUC_hi=pct(da)[1],
                              d_rmse=r0[m] - r0[ref], d_rmse_lo=pct(dr)[0], d_rmse_hi=pct(dr)[1],
                              P_model_more_consistent=float((dr < 0).mean())))
    pd.DataFrame(pairs).to_csv(os.path.join(rq2, "rq2_folds_pairs.csv"), index=False, float_format="%.5f")
    # per-fold point estimates
    rows = []
    for name, f in folds.items():
        wts = f["N"].sum(0)
        for m in MODELS:
            if m in f["S"]:
                N = f["N_dkt"] if m == "M7 DKT-time" else f["N"]
                curve = np.divide(f["S"][m].sum(0), N.sum(0), out=np.zeros(len(LAGS)), where=N.sum(0) > 0)
            else:
                curve = np.zeros(len(LAGS))
            rows.append(dict(fold=name, model=m, AUC=roc_auc_score(f["y"], f["preds"][m]),
                             rmse_lag=np.sqrt(np.average((curve - f["mu"]) ** 2, weights=wts)),
                             causal_all=f["mu"] @ wts / wts.sum(), implied_all=curve @ wts / wts.sum()))
    pd.DataFrame(rows).to_csv(os.path.join(rq2, "rq2_folds_by_fold.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 240)
    sh = summ.copy()
    for c in ("implied_all", "implied_lo", "implied_hi", "rmse_lag", "rmse_lo", "rmse_hi", "causal_all_pooled"):
        sh[c] = (sh[c] * 100).round(2)
    print(sh.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pp = pd.DataFrame(pairs)
    for c in ("d_rmse", "d_rmse_lo", "d_rmse_hi"):
        pp[c] = (pp[c] * 100).round(2)
    print(pp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 500)
