"""
Second-review checks on the pooled three-window comparison (H2):

    python review_pooled_variants.py <rq1_dir> <rq2_dir> <leakfree_root> <das3h_direct_root> <out_dir> [n_boot]

Variants of the inputs of rq2_bootstrap_folds.py (same loader, learner-clustered bootstrap, causal references
redrawn independently per window, spec B for all re-exposures and spec C after correct / after wrong answers):
  main        as in the paper (online Elo updated per response; DAS3H full-history deletion)
  leakfree    GBM and GBM without memory retrained with Elo updated after each game (review_rq2_leakfree.py)
  das3h_direct  DAS3H with item windows, direct item-memory protocol (review_das3h_direct.py)
Outputs pooled AUC and causal inconsistency D_m per model and group, and the key paired differences.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_bootstrap_folds import KEY, LAGS, MODELS, load_fold  # noqa: E402
from review_pigeonhole import ref_C  # noqa: E402

SEED = 42
GROUPS = ("all", "after correct", "after error")
TAGS = {"F1": "2025-07", "F2": "2025-10", "F3": ""}


def sub(root, tag):
    return root if tag == "" else os.path.join(root, "fold_" + tag)


def apply_variant(f, tag, variant, leak_root, d3_root):
    b = f["base"]
    if variant == "leakfree":
        lf = pd.read_parquet(os.path.join(sub(leak_root, tag), "rq2_test_predictions.parquet"))
        lf = lf[~lf.duplicated(KEY, keep=False)]
        m = b[KEY].merge(lf[KEY + ["M5 GBM (all features)", "M5 GBM (all features) | cf unseen", "M5a GBM without memory features"]],
                         on=KEY, how="left")
        assert m["M5 GBM (all features)"].notna().all()
        f["preds"]["M5 GBM (all features)"] = m["M5 GBM (all features)"].values
        f["preds"]["M5a GBM without memory features"] = m["M5a GBM without memory features"].values
        f["implied"]["M5 GBM (all features)"] = (m["M5 GBM (all features)"] - m["M5 GBM (all features) | cf unseen"]).values
    if variant == "das3h_direct":
        d3 = pd.read_parquet(os.path.join(sub(d3_root, tag), "rq2_rows_das3h.parquet"))
        d3 = d3[~d3.duplicated(KEY, keep=False)]
        m = b[KEY].merge(d3[KEY + ["implied | M6b DAS3H + item windows"]], on=KEY, how="left")
        assert m["implied | M6b DAS3H + item windows"].notna().all()
        f["implied"]["M6b DAS3H + item windows"] = m["implied | M6b DAS3H + item windows"].values
    return f


def run(rq1, rq2, variant, leak_root, d3_root, n_boot):
    rng = np.random.default_rng(SEED)
    folds = {k: apply_variant(load_fold(rq1, rq2, t), t, variant, leak_root, d3_root) for k, t in TAGS.items()}
    uids = pd.Index(np.unique(np.concatenate([f["base"].uid.values for f in folds.values()])))
    U = len(uids)
    lag_ix = {l: j for j, l in enumerate(LAGS)}
    for k, f in folds.items():
        b = f["base"]
        f["u"], f["y"] = uids.get_indexer(b.uid), b.y.values
        f["seen"] = b.exposure.values > 1
        f["lag"] = np.array([lag_ix.get(x, -1) for x in b.lag_bin.values])
        f["prev"] = b.prev_y.values
        r = f["dkt_rows"]
        f["dkt"] = dict(u=uids.get_indexer(r.uid), imp=r.implied.values, lag=np.array([lag_ix.get(x, -1) for x in r.lag_bin.values]),
                        prev=r.prev_y.values)
        f["refC"] = ref_C(rq1, TAGS[k])
        f["drawsB"] = rng.multivariate_normal(f["mu"], f["V"], size=n_boot)
        f["drawsC"] = {g: rng.multivariate_normal(m, V, size=n_boot) for g, (m, V) in f["refC"].items()}

    def curve(w, lag, imp, sel):
        s = np.bincount(lag[sel], w[sel] * imp[sel], minlength=len(LAGS))
        n = np.bincount(lag[sel], w[sel], minlength=len(LAGS))
        return np.divide(s, n, out=np.zeros(len(LAGS)), where=n > 0)

    def gsel(prev, g):
        return np.ones(len(prev), bool) if g == "all" else prev == (1 if g == "after correct" else 0)

    def stats(wu, b):
        auc = {m: [] for m in MODELS}; size = []
        sq = {(m, g): 0.0 for m in MODELS for g in GROUPS}; wsum = {g: 0.0 for g in GROUPS}
        for f in folds.values():
            w = wu[f["u"]]; keep = w > 0; size.append(w.sum())
            for m in MODELS:
                auc[m].append(roc_auc_score(f["y"][keep], f["preds"][m][keep], sample_weight=w[keep]))
            for g in GROUPS:
                sel = f["seen"] & (f["lag"] >= 0) & gsel(f["prev"], g)
                wts = np.bincount(f["lag"][sel], w[sel], minlength=len(LAGS))
                ref = (f["mu"] if b is None else f["drawsB"][b]) if g == "all" else (f["refC"][g][0] if b is None else f["drawsC"][g][b])
                wsum[g] += wts.sum()
                for m in MODELS:
                    if m == "M7 DKT-time":
                        d = f["dkt"]; c = curve(wu[d["u"]], d["lag"], d["imp"], (d["lag"] >= 0) & gsel(d["prev"], g))
                    elif m in f["implied"]:
                        c = curve(w, f["lag"], f["implied"][m], sel)
                    else:
                        c = np.zeros(len(LAGS))
                    sq[(m, g)] += (wts * (c - ref) ** 2).sum()
        size = np.array(size)
        return {m: np.average(auc[m], weights=size) for m in MODELS}, {k: np.sqrt(v / wsum[k[1]]) for k, v in sq.items()}

    a0, r0 = stats(np.ones(U), None)
    A = np.zeros((n_boot, len(MODELS))); R = {k: np.zeros(n_boot) for k in r0}
    for bb in range(n_boot):
        wu = rng.multinomial(U, np.full(U, 1 / U)).astype(float)
        a, r = stats(wu, bb)
        A[bb] = [a[m] for m in MODELS]
        for k in r:
            R[k][bb] = r[k]
    pct = lambda x: np.percentile(x, [2.5, 97.5])
    lev = pd.DataFrame([dict(variant=variant, model=m, group=g, AUC=a0[m], D_pp=100 * r0[(m, g)],
                             lo=100 * pct(R[(m, g)])[0], hi=100 * pct(R[(m, g)])[1]) for m in MODELS for g in GROUPS])
    rows = []
    for ref, m in (("M5 GBM (all features)", "M5a GBM without memory features"), ("M5 GBM (all features)", "M6b DAS3H + item windows"),
                   ("M5 GBM (all features)", "M7 DKT-time")):
        da = A[:, MODELS.index(m)] - A[:, MODELS.index(ref)]
        row = dict(variant=variant, reference=ref, model=m, dAUC=a0[m] - a0[ref], dAUC_lo=pct(da)[0], dAUC_hi=pct(da)[1])
        for g in GROUPS:
            dr = R[(m, g)] - R[(ref, g)]
            row.update({f"dD_{g}": 100 * (r0[(m, g)] - r0[(ref, g)]), f"dD_{g}_lo": 100 * pct(dr)[0], f"dD_{g}_hi": 100 * pct(dr)[1]})
        rows.append(row)
    return lev, pd.DataFrame(rows)


def main(rq1, rq2, leak_root, d3_root, out, n_boot=500):
    os.makedirs(out, exist_ok=True)
    levs, pairs = [], []
    for v in ("main", "leakfree", "das3h_direct"):
        lev, pr = run(rq1, rq2, v, leak_root, d3_root, n_boot)
        levs.append(lev); pairs.append(pr)
        print(v, "done", flush=True)
    lev, pr = pd.concat(levs), pd.concat(pairs)
    lev.to_csv(os.path.join(out, "pooled_variants_levels.csv"), index=False, float_format="%.5f")
    pr.to_csv(os.path.join(out, "pooled_variants_pairs.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 260)
    print(lev.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(*sys.argv[1:6], int(sys.argv[6]) if len(sys.argv) > 6 else 500)
