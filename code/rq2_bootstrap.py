"""
RQ2: learner-clustered bootstrap for AUC, AUC differences, implied curves and causal consistency.

    python rq2_bootstrap.py <rq1_dir> <rq2_dir> [n_boot]

Inputs (all on the RQ2 test window 2026-01-01 .. 2026-03-31):
  rq2_test_predictions.parquet          M0-M5a predictions (+ 'cf unseen' for M3, M5)
  rq2_rows_das3h.parquet                M6, M6b predictions and implied effects
  rq2_rows_deep_<name>.parquet          M7 predictions; rq2_implied_rows_deep_<name>.parquet implied (40k sample)
  rq1_effects_test.csv, rq1_cov_test_B.npy (+ _terms.csv)   causal lag curve and its cluster-robust covariance
Bootstrap: learners resampled with replacement (multinomial weights), n_boot replicates, seed 42.
Each replicate: AUC per model (frequency-weighted), implied lag curve per model, causal curve drawn
from N(coef, V); rmse_lag = sqrt(weighted mean over lag bins of (implied - causal)^2), weights = the
model's re-exposures per bin in the replicate. Models without an item-specific path have implied = 0.
Rows are matched across models by (uid, game_id, question_id); keys that occur twice (an item listed
twice in one game) are dropped. Outputs: rq2_bootstrap_summary.csv, rq2_bootstrap_pairs.csv.
"""
import glob
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


def main(rq1, rq2, n_boot=500):
    rng = np.random.default_rng(SEED)
    # ---------- predictions on the common test rows ----------
    s1 = pd.read_parquet(os.path.join(rq2, "rq2_test_predictions.parquet"))
    s1["lag_bin"] = lag_bin(np.where(s1.exposure > 1, np.expm1(s1.log_gap_h), -1))
    base = s1[KEY + ["y", "exposure", "lag_bin"]].copy()
    preds, implied = {}, {}
    for col in s1.columns:
        if col.startswith("M") and "| cf" not in col:
            preds[col] = s1[col].values
            if col + " | cf unseen" in s1.columns:
                implied[col] = s1[col].values - s1[col + " | cf unseen"].values
    d3 = pd.read_parquet(os.path.join(rq2, "rq2_rows_das3h.parquet"))
    base = base.merge(d3[KEY + [c for c in d3.columns if c.startswith(("p |", "implied |"))]], on=KEY, how="inner")
    for f in sorted(glob.glob(os.path.join(rq2, "rq2_rows_deep_*.parquet"))):
        name = "M7 " + os.path.basename(f)[len("rq2_rows_deep_"):-len(".parquet")]
        dd = pd.read_parquet(f).rename(columns={"p": "p | " + name})
        base = base.merge(dd[KEY + ["p | " + name]], on=KEY, how="inner")
    dup = base.duplicated(KEY, keep=False)
    base = base[~dup].reset_index(drop=True)
    s1 = s1.merge(base[KEY], on=KEY, how="inner")
    s1 = s1[~s1.duplicated(KEY, keep=False)].set_index(KEY).loc[base.set_index(KEY).index].reset_index()
    for col in list(preds):
        preds[col] = s1[col].values
        if col in implied:
            implied[col] = s1[col].values - s1[col + " | cf unseen"].values
    for c in base.columns:
        if c.startswith("p | "):
            preds[c[4:]] = base[c].values
        if c.startswith("implied | "):
            implied[c[10:]] = base[c].values
    print("common test rows:", len(base), "dropped duplicate keys:", int(dup.sum()), flush=True)

    y = base.y.values
    u_codes, u_uniq = pd.factorize(base.uid)
    U = len(u_uniq)
    seen = base.exposure.values > 1
    lb = base.lag_bin.values

    # per-learner x lag-bin sums of implied effects and counts (row-level models)
    def learner_bin_sums(imp, mask_rows, u_idx, bins):
        S = np.zeros((U, len(LAGS))); N = np.zeros((U, len(LAGS)))
        for j, lab in enumerate(LAGS):
            m = mask_rows & (bins == lab)
            S[:, j] = np.bincount(u_idx[m], imp[m], minlength=U)
            N[:, j] = np.bincount(u_idx[m], minlength=U)
        return S, N

    curves = {}
    for name in preds:
        if name in implied:
            curves[name] = learner_bin_sums(implied[name], seen, u_codes, lb)
    for f in sorted(glob.glob(os.path.join(rq2, "rq2_implied_rows_deep_*.parquet"))):
        name = "M7 " + os.path.basename(f)[len("rq2_implied_rows_deep_"):-len(".parquet")]
        r = pd.read_parquet(f)
        ui = pd.Index(u_uniq).get_indexer(r.uid)
        ok = ui >= 0
        curves[name] = learner_bin_sums(r.implied.values[ok], np.ones(ok.sum(), bool), ui[ok], r.lag_bin.values[ok])
    _, N_ref = learner_bin_sums(np.zeros(len(base)), seen, u_codes, lb)

    # causal lag curve and covariance
    ce = pd.read_csv(os.path.join(rq1, "rq1_effects_test.csv"))
    ce = ce[(ce.spec == "B lag") & ce.estimator.str.startswith("E1")].set_index("term")
    terms = pd.read_csv(os.path.join(rq1, "rq1_cov_test_B_terms.csv")).iloc[:, 0].tolist()
    V = np.load(os.path.join(rq1, "rq1_cov_test_B.npy"))
    ix = [terms.index("lag " + l) for l in LAGS]
    mu, Vl = ce.loc[["lag " + l for l in LAGS], "coef"].values, V[np.ix_(ix, ix)]
    causal_draws = rng.multivariate_normal(mu, Vl, size=n_boot)

    names = list(preds)
    auc = np.zeros((n_boot, len(names))); rmse = np.zeros((n_boot, len(names)))
    imp_all = np.zeros((n_boot, len(names)))
    for b in range(n_boot):
        w_u = rng.multinomial(U, np.full(U, 1 / U)).astype(float)
        w = w_u[u_codes]
        keep = w > 0
        for k, name in enumerate(names):
            auc[b, k] = roc_auc_score(y[keep], preds[name][keep], sample_weight=w[keep])
            if name in curves:
                S, N = curves[name]
                s_b, n_b = w_u @ S, w_u @ N
                curve = np.divide(s_b, n_b, out=np.zeros_like(s_b), where=n_b > 0)
                wts = w_u @ N_ref
                imp_all[b, k] = s_b.sum() / max(n_b.sum(), 1)
            else:
                curve = np.zeros(len(LAGS)); wts = w_u @ N_ref
            rmse[b, k] = np.sqrt(np.average((curve - causal_draws[b]) ** 2, weights=wts))
        if (b + 1) % 50 == 0:
            print("replicate", b + 1, flush=True)

    pct = lambda a: np.percentile(a, [2.5, 97.5], axis=0)
    a_lo, a_hi = pct(auc); r_lo, r_hi = pct(rmse); i_lo, i_hi = pct(imp_all)
    best_auc = np.bincount(auc.argmax(1), minlength=len(names)) / n_boot
    best_cons = np.bincount(rmse.argmin(1), minlength=len(names)) / n_boot
    summ = pd.DataFrame({"model": names, "AUC": auc.mean(0), "AUC_lo": a_lo, "AUC_hi": a_hi,
                         "implied_all": imp_all.mean(0), "implied_lo": i_lo, "implied_hi": i_hi,
                         "rmse_lag": rmse.mean(0), "rmse_lo": r_lo, "rmse_hi": r_hi,
                         "P(best AUC)": best_auc, "P(best consistency)": best_cons})
    summ.to_csv(os.path.join(rq2, "rq2_bootstrap_summary.csv"), index=False, float_format="%.5f")
    pairs = []
    for ref in ("M5 GBM (all features)", "M6b DAS3H + item windows"):
        if ref not in names:
            continue
        r = names.index(ref)
        for k, name in enumerate(names):
            if k == r:
                continue
            da, dr = auc[:, k] - auc[:, r], rmse[:, k] - rmse[:, r]
            pairs.append(dict(reference=ref, model=name, dAUC=da.mean(), dAUC_lo=pct(da)[0], dAUC_hi=pct(da)[1],
                              d_rmse=dr.mean(), d_rmse_lo=pct(dr)[0], d_rmse_hi=pct(dr)[1]))
    pd.DataFrame(pairs).to_csv(os.path.join(rq2, "rq2_bootstrap_pairs.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 240)
    show = summ.copy()
    for c in ("implied_all", "implied_lo", "implied_hi", "rmse_lag", "rmse_lo", "rmse_hi"):
        show[c] = (show[c] * 100).round(2)
    print(show.sort_values("AUC", ascending=False).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pp = pd.DataFrame(pairs)
    for c in ("d_rmse", "d_rmse_lo", "d_rmse_hi"):
        pp[c] = (pp[c] * 100).round(2)
    print(pp.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 500)
