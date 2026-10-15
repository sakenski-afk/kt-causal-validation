"""
Reviewer check: two-way (learner x item) 'pigeonhole' bootstrap of the RQ2 model comparison over the three test windows.

    python review_pigeonhole.py <rq1_dir> <rq2_dir> <out_dir> [n_boot]

Same data, models and pooled statistics as rq2_bootstrap_folds.py (reused loader), but each replicate draws
multinomial weights for learners AND for items and weights every response by w_u * w_i, so that dependence through
shared items (many learners, both duel partners) is propagated. Reported for the aggregate lag curve (spec B reference)
and separately after a correct and after a wrong previous answer (spec C reference), on the probability scale with
re-exposure weights. The causal reference is redrawn from its cluster-robust sampling distribution as in the paper.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_bootstrap_folds import LAGS, MODELS, load_fold  # noqa: E402

SEED = 42
GROUPS = ("all", "after correct", "after error")


def ref_C(rq1, tag):
    sfx = "_test" + ("" if tag == "" else "_" + tag)
    ce = pd.read_csv(os.path.join(rq1, f"rq1_effects{sfx}.csv"))
    ce = ce[(ce.spec == "C lag x prev") & ce.estimator.str.startswith("E1")].set_index("term")
    terms = pd.read_csv(os.path.join(rq1, f"rq1_cov{sfx}_C_terms.csv")).iloc[:, 0].tolist()
    V = np.load(os.path.join(rq1, f"rq1_cov{sfx}_C.npy"))
    out = {}
    for g in ("after correct", "after error"):
        names = [f"{g}, lag {l}" for l in LAGS]
        ix = [terms.index(n) for n in names]
        out[g] = (ce.loc[names, "coef"].values, V[np.ix_(ix, ix)])
    return out


def main(rq1, rq2, out, n_boot=500):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    tags = {"F1": "2025-07", "F2": "2025-10", "F3": ""}
    folds = {k: load_fold(rq1, rq2, t) for k, t in tags.items()}
    for k, t in tags.items():
        folds[k]["refC"] = ref_C(rq1, t)
    uids = pd.Index(np.unique(np.concatenate([f["base"].uid.values for f in folds.values()])))
    items = pd.Index(np.unique(np.concatenate([f["base"].question_id.values for f in folds.values()])))
    U, I = len(uids), len(items)
    lag_ix = {l: j for j, l in enumerate(LAGS)}
    for f in folds.values():
        b = f["base"]
        f["u"], f["i"], f["y"] = uids.get_indexer(b.uid), items.get_indexer(b.question_id), b.y.values
        f["seen"] = b.exposure.values > 1
        f["lag"] = np.array([lag_ix.get(x, -1) for x in b.lag_bin.values])
        f["prev"] = b.prev_y.values
        r = f["dkt_rows"]
        f["dkt"] = dict(u=uids.get_indexer(r.uid), i=items.get_indexer(r.question_id), imp=r.implied.values,
                        lag=np.array([lag_ix.get(x, -1) for x in r.lag_bin.values]), prev=r.prev_y.values)
        f["drawsB"] = rng.multivariate_normal(f["mu"], f["V"], size=n_boot)
        f["drawsC"] = {g: rng.multivariate_normal(m, V, size=n_boot) for g, (m, V) in f["refC"].items()}

    def curve(w, lag, imp, sel):
        s = np.bincount(lag[sel], w[sel] * imp[sel], minlength=len(LAGS))
        n = np.bincount(lag[sel], w[sel], minlength=len(LAGS))
        return np.divide(s, n, out=np.zeros(len(LAGS)), where=n > 0)

    def stats(wu, wi, b):
        auc = {m: [] for m in MODELS}; size = []
        sq = {(m, g): 0.0 for m in MODELS for g in GROUPS}; wsum = {g: 0.0 for g in GROUPS}
        for f in folds.values():
            w = wu[f["u"]] * wi[f["i"]]
            keep = w > 0
            size.append(w.sum())
            for m in MODELS:
                auc[m].append(roc_auc_score(f["y"][keep], f["preds"][m][keep], sample_weight=w[keep]))
            for g in GROUPS:
                sel = f["seen"] & (f["lag"] >= 0) & (True if g == "all" else (f["prev"] == (1 if g == "after correct" else 0)))
                wts = np.bincount(f["lag"][sel], w[sel], minlength=len(LAGS))
                ref = (f["mu"] if b is None else f["drawsB"][b]) if g == "all" else \
                      (f["refC"][g][0] if b is None else f["drawsC"][g][b])
                wsum[g] += wts.sum()
                for m in MODELS:
                    if m == "M7 DKT-time":
                        d = f["dkt"]; wd = wu[d["u"]] * wi[d["i"]]
                        sd = (d["lag"] >= 0) & (True if g == "all" else (d["prev"] == (1 if g == "after correct" else 0)))
                        c = curve(wd, d["lag"], d["imp"], sd)
                    elif m in f["implied"]:
                        c = curve(w, f["lag"], f["implied"][m], sel)
                    else:
                        c = np.zeros(len(LAGS))
                    sq[(m, g)] += (wts * (c - ref) ** 2).sum()
        size = np.array(size)
        return ({m: np.average(auc[m], weights=size) for m in MODELS},
                {k: np.sqrt(v / wsum[k[1]]) for k, v in sq.items()})

    a0, r0 = stats(np.ones(U), np.ones(I), None)
    A = np.zeros((n_boot, len(MODELS))); R = {k: np.zeros(n_boot) for k in r0}
    for b in range(n_boot):
        wu = rng.multinomial(U, np.full(U, 1 / U)).astype(float)
        wi = rng.multinomial(I, np.full(I, 1 / I)).astype(float)
        a, r = stats(wu, wi, b)
        A[b] = [a[m] for m in MODELS]
        for k in r:
            R[k][b] = r[k]
        if (b + 1) % 50 == 0:
            print("replicate", b + 1, flush=True)
    pct = lambda x: np.percentile(x, [2.5, 97.5])
    rows = []
    for ref, m in (("M5 GBM (all features)", "M6b DAS3H + item windows"), ("M5 GBM (all features)", "M5a GBM without memory features"),
                   ("M5 GBM (all features)", "M7 DKT-time"), ("M6b DAS3H + item windows", "M7 DKT-time")):
        da = A[:, MODELS.index(m)] - A[:, MODELS.index(ref)]
        row = dict(reference=ref, model=m, dAUC=a0[m] - a0[ref], dAUC_lo=pct(da)[0], dAUC_hi=pct(da)[1])
        for g in GROUPS:
            dr = R[(m, g)] - R[(ref, g)]
            row.update({f"dD_{g}_pp": 100 * (r0[(m, g)] - r0[(ref, g)]), f"dD_{g}_lo": 100 * pct(dr)[0],
                        f"dD_{g}_hi": 100 * pct(dr)[1], f"P_model_better_{g}": float((dr < 0).mean())})
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "pigeonhole_pairs.csv"), index=False, float_format="%.5f")
    lev = pd.DataFrame([dict(model=m, group=g, D_pp=100 * r0[(m, g)], lo=100 * pct(R[(m, g)])[0], hi=100 * pct(R[(m, g)])[1])
                        for m in MODELS for g in GROUPS])
    lev.to_csv(os.path.join(out, "pigeonhole_levels.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 250)
    print(lev.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 500)
