"""
RQ2: full predictive-metric comparison (AUC, log-loss, Brier RMSE, ECE) with learner-clustered bootstrap CIs.

    python rq2_metrics_bootstrap.py <rq2_dir> [n_boot]

(1) Test window F3 (2026-01..03): all models of Table 5 (M1-M7; predictions of rq2_models.py,
    rq2_das3h.py, rq2_deep.py), matched by (uid, game_id, question_id).
(2) Three rolling test windows pooled: M5 GBM, M5a GBM without memory, M6b DAS3H + item windows, M7 DKT-time
    (metric = test-size-weighted mean of the window metrics; learners resampled once across windows).
Metrics on all test responses; ECE also on re-exposures only (a model without item memory can be calibrated
overall yet miscalibrated exactly where re-exposure effects live). ECE: 10 equal-width bins.
Pairwise differences (model - reference) for GBM vs GBM w/o memory, GBM vs DAS3H, GBM vs DKT-time, DAS3H vs DKT-time.
Also writes reliability-diagram data (20 equal-width bins) for Figure 6.
Outputs: rq2_metrics_full_F3.csv, rq2_metrics_full_pooled.csv, rq2_metrics_pairs.csv, rq2_calibration_bins.csv.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SEED = 42
KEY = ["uid", "game_id", "question_id"]
EPS = 1e-6
PAIRS = [("M5 GBM (all features)", "M5a GBM without memory features"), ("M5 GBM (all features)", "M6b DAS3H + item windows"),
         ("M5 GBM (all features)", "M7 DKT-time"), ("M6b DAS3H + item windows", "M7 DKT-time")]
POOLED = ["M5 GBM (all features)", "M5a GBM without memory features", "M6b DAS3H + item windows", "M7 DKT-time"]


def load_window(d, deep=("DKT-time", "DKT", "SAKT"), stage1_models=None):
    s1 = pd.read_parquet(os.path.join(d, "rq2_test_predictions.parquet"))
    cols = [c for c in s1.columns if c.startswith("M") and "| cf" not in c and (stage1_models is None or c in stage1_models)]
    base = s1[KEY + ["y", "exposure"] + cols]
    d3 = pd.read_parquet(os.path.join(d, "rq2_rows_das3h.parquet"))
    pc = [c for c in d3.columns if c.startswith("p | ")]
    base = base.merge(d3[KEY + pc].rename(columns={c: c[4:] for c in pc}), on=KEY)
    for name in deep:
        f = os.path.join(d, f"rq2_rows_deep_{name}.parquet")
        if os.path.exists(f):
            dd = pd.read_parquet(f).rename(columns={"p": f"M7 {name}"})
            base = base.merge(dd[KEY + [f"M7 {name}"]], on=KEY)
    base = base[~base.duplicated(KEY, keep=False)].reset_index(drop=True)
    models = [c for c in base.columns if c.startswith("M") and c != "M0 Global mean"]
    return base, models


def wmetrics(y, p, w, seen):
    p = np.clip(p, EPS, 1 - EPS)
    keep = w > 0
    y, p, w, seen = y[keep], p[keep], w[keep], seen[keep]
    auc = roc_auc_score(y, p, sample_weight=w)
    ll = -np.sum(w * (y * np.log(p) + (1 - y) * np.log(1 - p))) / w.sum()
    rmse = np.sqrt(np.sum(w * (y - p) ** 2) / w.sum())

    def ece(mask):
        b = np.minimum((p[mask] * 10).astype(int), 9)
        ww, yy, pp = w[mask], y[mask], p[mask]
        sy, sp, sw = (np.bincount(b, ww * yy, 10), np.bincount(b, ww * pp, 10), np.bincount(b, ww, 10))
        return np.sum(np.abs(sy - sp)) / sw.sum()
    return np.array([auc, ll, rmse, ece(np.ones(len(y), bool)), ece(seen)])


NAMES = ["AUC", "LogLoss", "RMSE", "ECE", "ECE_reexposures"]


def main(rq2, n_boot=500):
    rng = np.random.default_rng(SEED)
    pct = lambda a: np.percentile(a, [2.5, 97.5], axis=0)
    # ---------------- (1) F3, all models ----------------
    base, models = load_window(rq2)
    y, seen = base.y.values.astype(float), base.exposure.values > 1
    u = pd.factorize(base.uid)[0]; U = u.max() + 1
    point = {m: wmetrics(y, base[m].values, np.ones(len(y)), seen) for m in models}
    boot = {m: np.zeros((n_boot, len(NAMES))) for m in models}
    for b in range(n_boot):
        w = rng.multinomial(U, np.full(U, 1 / U)).astype(float)[u]
        for m in models:
            boot[m][b] = wmetrics(y, base[m].values, w, seen)
        if (b + 1) % 100 == 0:
            print("F3 replicate", b + 1, flush=True)
    rows = []
    for m in models:
        lo, hi = pct(boot[m])
        rows.append(dict(model=m, **{k: point[m][i] for i, k in enumerate(NAMES)},
                         **{k + "_lo": lo[i] for i, k in enumerate(NAMES)}, **{k + "_hi": hi[i] for i, k in enumerate(NAMES)}))
    pd.DataFrame(rows).sort_values("AUC", ascending=False).to_csv(os.path.join(rq2, "rq2_metrics_full_F3.csv"), index=False, float_format="%.5f")
    pairs = []
    for a, b_ in PAIRS:
        if a in boot and b_ in boot:
            d = boot[b_] - boot[a]
            lo, hi = pct(d)
            pairs.append(dict(window="F3", reference=a, model=b_, **{f"d{k}": point[b_][i] - point[a][i] for i, k in enumerate(NAMES)},
                              **{f"d{k}_lo": lo[i] for i, k in enumerate(NAMES)}, **{f"d{k}_hi": hi[i] for i, k in enumerate(NAMES)}))
    # reliability bins (F3)
    cal = []
    for m in models:
        p = np.clip(base[m].values, EPS, 1 - EPS)
        for subset, mask in (("all", np.ones(len(y), bool)), ("re-exposures", seen)):
            bins = np.minimum((p[mask] * 20).astype(int), 19)
            for k in range(20):
                sel = bins == k
                if sel.sum() >= 200:
                    cal.append(dict(model=m, subset=subset, bin=k, mean_pred=p[mask][sel].mean(), obs=y[mask][sel].mean(), n=int(sel.sum())))
    pd.DataFrame(cal).to_csv(os.path.join(rq2, "rq2_calibration_bins.csv"), index=False, float_format="%.5f")

    # ---------------- (2) three windows pooled ----------------
    wins = []
    for sub in ("fold_2025-07", "fold_2025-10", ""):
        d = os.path.join(rq2, sub) if sub else rq2
        bw, _ = load_window(d, deep=("DKT-time",), stage1_models=POOLED[:2])
        wins.append(bw)
    uids = pd.Index(np.unique(np.concatenate([w_.uid.values for w_ in wins])))
    UU = len(uids)
    wins = [(w_, uids.get_indexer(w_.uid)) for w_ in wins]
    sizes = np.array([len(w_) for w_, _ in wins], float)

    def pooled(wu):
        out = {}
        for m in POOLED:
            vals = np.array([wmetrics(w_.y.values.astype(float), w_[m].values, wu[ix], w_.exposure.values > 1) for w_, ix in wins])
            fs = np.array([wu[ix].sum() for _, ix in wins])
            out[m] = np.average(vals, axis=0, weights=fs)
        return out
    p0 = pooled(np.ones(UU))
    pb = {m: np.zeros((n_boot, len(NAMES))) for m in POOLED}
    for b in range(n_boot):
        wu = rng.multinomial(UU, np.full(UU, 1 / UU)).astype(float)
        r = pooled(wu)
        for m in POOLED:
            pb[m][b] = r[m]
        if (b + 1) % 100 == 0:
            print("pooled replicate", b + 1, flush=True)
    rows = []
    for m in POOLED:
        lo, hi = pct(pb[m])
        rows.append(dict(model=m, **{k: p0[m][i] for i, k in enumerate(NAMES)},
                         **{k + "_lo": lo[i] for i, k in enumerate(NAMES)}, **{k + "_hi": hi[i] for i, k in enumerate(NAMES)}))
    pd.DataFrame(rows).to_csv(os.path.join(rq2, "rq2_metrics_full_pooled.csv"), index=False, float_format="%.5f")
    for a, b_ in PAIRS:
        d = pb[b_] - pb[a]
        lo, hi = pct(d)
        pairs.append(dict(window="pooled", reference=a, model=b_, **{f"d{k}": p0[b_][i] - p0[a][i] for i, k in enumerate(NAMES)},
                          **{f"d{k}_lo": lo[i] for i, k in enumerate(NAMES)}, **{f"d{k}_hi": hi[i] for i, k in enumerate(NAMES)}))
    pd.DataFrame(pairs).to_csv(os.path.join(rq2, "rq2_metrics_pairs.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 250)
    print(pd.read_csv(os.path.join(rq2, "rq2_metrics_full_F3.csv"))[["model"] + NAMES].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.read_csv(os.path.join(rq2, "rq2_metrics_full_pooled.csv"))[["model"] + NAMES].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pp = pd.DataFrame(pairs)
    print(pp[["window", "reference", "model"] + [c for c in pp.columns if c.startswith("d")]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 500)
