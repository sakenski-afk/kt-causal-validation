"""
MAKE revision: AKT and HawkesKT added to the pooled three-window comparison (H2), primary specification.

    python review_modern_kt.py <rq1_dir> <rq2_dir> <modern_root> <out_dir> [n_boot]

<rq2_dir> = work_v2/rq2 (leak-free Elo, direct protocol; loaded by rq2_bootstrap_folds.load_fold).
<modern_root>/{F3, fold_2025-07, fold_2025-10} hold rq2_rows_deep_<m>.parquet and rq2_implied_rows_deep_<m>.parquet
from rq2_deep_modern.py (same seeded 40k re-exposure sample per window as DKT-time).
Same estimator as review_pooled_variants.py (variant 'main' on the work_v2 inputs): learner-clustered bootstrap with
weights shared across windows, causal references redrawn per window (spec B for all re-exposures, spec C after
correct / after wrong answers). The aggregate ('all') is computed in a separate run with the random stream of rq2_bootstrap_folds.py and the
groups with that of review_pooled_variants.py, so the original models reproduce the published values exactly.
Outputs: modern_levels.csv (AUC, D_m with CI per model and group), modern_pairs.csv (paired differences),
modern_curves.csv (pooled implied curves per model, group and lag bin).
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_bootstrap_folds import KEY, LAGS, MODELS as BASE_MODELS, load_fold  # noqa: E402
from review_pigeonhole import ref_C  # noqa: E402

SEED = 42
GROUPS = ("all", "after correct", "after error")
TAGS = {"F1": "2025-07", "F2": "2025-10", "F3": ""}
NEW = ["AKT", "HawkesKT"]


def run(rq1, rq2, modern, n_boot, new, groups=GROUPS):
    rng = np.random.default_rng(SEED)
    models = BASE_MODELS + [f"M7 {n}" for n in new]
    sample_models = ["M7 DKT-time"] + [f"M7 {n}" for n in new]
    folds = {}
    for k, t in TAGS.items():
        f = load_fold(rq1, rq2, t)
        md = os.path.join(modern, "F3" if t == "" else "fold_" + t)
        rows = {"M7 DKT-time": f["dkt_rows"]}
        for n in new:
            p = pd.read_parquet(os.path.join(md, f"rq2_rows_deep_{n}.parquet"))
            p = p[~p.duplicated(KEY, keep=False)]
            m = f["base"][KEY].merge(p[KEY + ["p"]], on=KEY, how="left")
            assert m.p.notna().all(), (k, n, int(m.p.isna().sum()))
            f["preds"][f"M7 {n}"] = m.p.values
            rows[f"M7 {n}"] = pd.read_parquet(os.path.join(md, f"rq2_implied_rows_deep_{n}.parquet"))
        f["rows"] = rows
        folds[k] = f
    uids = pd.Index(np.unique(np.concatenate([f["base"].uid.values for f in folds.values()])))
    U = len(uids)
    lag_ix = {l: j for j, l in enumerate(LAGS)}
    for k, f in folds.items():
        b = f["base"]
        f["u"], f["y"] = uids.get_indexer(b.uid), b.y.values
        f["seen"] = b.exposure.values > 1
        f["lag"] = np.array([lag_ix.get(x, -1) for x in b.lag_bin.values])
        f["prev"] = b.prev_y.values
        f["smp"] = {m: dict(u=uids.get_indexer(r.uid), imp=r.implied.values,
                            lag=np.array([lag_ix.get(x, -1) for x in r.lag_bin.values]), prev=r.prev_y.values)
                    for m, r in f["rows"].items()}
        f["refC"] = ref_C(rq1, TAGS[k])
        f["drawsB"] = rng.multivariate_normal(f["mu"], f["V"], size=n_boot)
        if len(groups) > 1:  # aggregate-only run reproduces the random stream of rq2_bootstrap_folds.py
            f["drawsC"] = {g: rng.multivariate_normal(m, V, size=n_boot) for g, (m, V) in f["refC"].items()}

    def curve(w, lag, imp, sel):
        s = np.bincount(lag[sel], w[sel] * imp[sel], minlength=len(LAGS))
        n = np.bincount(lag[sel], w[sel], minlength=len(LAGS))
        return np.divide(s, n, out=np.zeros(len(LAGS)), where=n > 0), s, n

    def gsel(prev, g):
        return np.ones(len(prev), bool) if g == "all" else prev == (1 if g == "after correct" else 0)

    def stats(wu, b, keep_curves=False):
        auc = {m: [] for m in models}; size = []
        sq = {(m, g): 0.0 for m in models for g in groups}; wsum = {g: 0.0 for g in groups}
        cs = {}
        for f in folds.values():
            w = wu[f["u"]]; keep = w > 0; size.append(w.sum())
            for m in models:
                auc[m].append(roc_auc_score(f["y"][keep], f["preds"][m][keep], sample_weight=w[keep]))
            for g in groups:
                sel = f["seen"] & (f["lag"] >= 0) & gsel(f["prev"], g)
                wts = np.bincount(f["lag"][sel], w[sel], minlength=len(LAGS))
                ref = (f["mu"] if b is None else f["drawsB"][b]) if g == "all" else (f["refC"][g][0] if b is None else f["drawsC"][g][b])
                wsum[g] += wts.sum()
                if keep_curves:
                    a = cs.setdefault(("CAUSAL", g), [np.zeros(len(LAGS)), np.zeros(len(LAGS))])
                    a[0] += wts * ref; a[1] += wts
                for m in models:
                    if m in sample_models:
                        d = f["smp"][m]; c, s, n = curve(wu[d["u"]], d["lag"], d["imp"], (d["lag"] >= 0) & gsel(d["prev"], g))
                    elif m in f["implied"]:
                        c, s, n = curve(w, f["lag"], f["implied"][m], sel)
                    else:
                        c, s, n = np.zeros(len(LAGS)), np.zeros(len(LAGS)), wts
                    sq[(m, g)] += (wts * (c - ref) ** 2).sum()
                    if keep_curves:
                        a = cs.setdefault((m, g), [np.zeros(len(LAGS)), np.zeros(len(LAGS))])
                        a[0] += s; a[1] += n
        size = np.array(size)
        out = ({m: np.average(auc[m], weights=size) for m in models}, {k: np.sqrt(v / wsum[k[1]]) for k, v in sq.items()})
        return out + (cs,) if keep_curves else out

    a0, r0, cs = stats(np.ones(U), None, keep_curves=True)
    A = np.zeros((n_boot, len(models))); R = {k: np.zeros(n_boot) for k in r0}
    for bb in range(n_boot):
        wu = rng.multinomial(U, np.full(U, 1 / U)).astype(float)
        a, r = stats(wu, bb)
        A[bb] = [a[m] for m in models]
        for k in r:
            R[k][bb] = r[k]
        if (bb + 1) % 100 == 0:
            print("replicate", bb + 1, flush=True)
    pct = lambda x: np.percentile(x, [2.5, 97.5])
    lev = pd.DataFrame([dict(model=m, group=g, AUC=a0[m], AUC_lo=pct(A[:, models.index(m)])[0], AUC_hi=pct(A[:, models.index(m)])[1],
                             D_pp=100 * r0[(m, g)], lo=100 * pct(R[(m, g)])[0], hi=100 * pct(R[(m, g)])[1],
                             P_best_D=float(np.mean(np.argmin(np.column_stack([R[(mm, g)] for mm in models]), 1) == models.index(m))))
                        for m in models for g in groups])
    rows = []
    refs = ("M5 GBM (all features)", "M6b DAS3H + item windows", "M7 DKT-time")
    for ref in refs:
        for m in models:
            if m == ref or (ref != "M5 GBM (all features)" and m not in [f"M7 {n}" for n in new]):
                continue
            da = A[:, models.index(m)] - A[:, models.index(ref)]
            row = dict(reference=ref, model=m, dAUC=a0[m] - a0[ref], dAUC_lo=pct(da)[0], dAUC_hi=pct(da)[1])
            for g in groups:
                dr = R[(m, g)] - R[(ref, g)]
                row.update({f"dD_{g}": 100 * (r0[(m, g)] - r0[(ref, g)]), f"dD_{g}_lo": 100 * pct(dr)[0],
                            f"dD_{g}_hi": 100 * pct(dr)[1], f"P_model_better_{g}": float((dr < 0).mean())})
            rows.append(row)
    cur = pd.DataFrame([dict(model=m, group=g, lag_bin=l, effect=s[j] / n[j] if n[j] else np.nan, n=n[j])
                        for (m, g), (s, n) in cs.items() for j, l in enumerate(LAGS)])
    return lev, pd.DataFrame(rows), cur


def main(rq1, rq2, modern, out, n_boot=500):
    os.makedirs(out, exist_ok=True)
    new = [n for n in NEW if all(os.path.exists(os.path.join(modern, d, f"rq2_implied_rows_deep_{n}.parquet"))
                                 for d in ("F3", "fold_2025-07", "fold_2025-10"))]
    print("new models with all three windows:", new, flush=True)
    lev, pr, cur = run(rq1, rq2, modern, n_boot, new)
    # aggregate ('all') as published: same random stream as rq2_bootstrap_folds.py
    lev_a, pr_a, _ = run(rq1, rq2, modern, n_boot, new, groups=("all",))
    lev = pd.concat([lev_a, lev[lev.group != "all"]], ignore_index=True)
    pr = pr_a.merge(pr.drop(columns=[c for c in pr.columns if "_all" in c or c.startswith("dAUC")]), on=["reference", "model"])
    lev.to_csv(os.path.join(out, "modern_levels.csv"), index=False, float_format="%.5f")
    pr.to_csv(os.path.join(out, "modern_pairs.csv"), index=False, float_format="%.5f")
    cur.to_csv(os.path.join(out, "modern_curves.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 300)
    print(lev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print((cur.pivot_table(index=["group", "lag_bin"], columns="model", values="effect", sort=False) * 100).round(1).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5], int(sys.argv[5]) if len(sys.argv) > 5 else 500)
