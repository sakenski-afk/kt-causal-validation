"""
RQ3: consequences of model choice for a spaced-review policy, evaluated against the causal curve.

    python rq3_policy.py <rq1_dir> <rq2_dir> [n_draws] [--asym-bin K]
    --fine: causal curve from the whole period (rq1_effects_fine.csv: 14 lag bins, 10 min .. >90 d; the shape of
    the curve is a population property, so using all 2.57 M responses is legitimate) instead of the test window;
    outputs get the suffix _fine.
    --models-dir DIR: read the model curves (rq2_lagshift_*.csv) from a rolling-origin fold directory instead of
    <rq2_dir>; outputs are written there (robustness of RQ3 across folds).
    --asym-bin: index of the lag point used as the asymptote of the normalization (default -1 = the
    longest lag, about 220 days; -2 = the 30-90 day bin). Outputs get the suffix _asym<K> when K != -1.

Decision: after a learner's encounter with an item, after how many hours should the item be shown again?
Policy of model m ("half-life review"): show the item again when the model-implied benefit of the earlier
encounter has decayed to half of its initial value,
    r(h) = (e(h) - e(h_max)) / (e(h_min) - e(h_min.. asymptote))  ->  H_m : r_m(H_m) = 0.5
with e(h) the implied effect of one earlier encounter h hours ago (rq2_lagshift_*.csv; h_min = 0.25 h,
asymptote = value at the longest lag). The same functional is computed from the causal lag curve
(rq1_effects_test.csv, spec C, E1, test window), separately after a correct and after an incorrect
earlier answer. Within each group the lag is quasi-randomized (RQ1), so the SHAPE of the causal curve is
causal even though its level after a previous error/correct answer is conditional; the half-life uses the
shape only. Causal curve: bin estimates placed at the geometric-mean lag of each bin (test re-exposures),
made non-increasing by isotonic regression, interpolated linearly in log(hours).
Policy evaluation against the causal curve:
    H_causal                 causal half-life
    retained_at_review       causal r(H_m): share of the benefit actually left when model m schedules the review (target 0.5)
    review_rate_ratio        H_causal / H_m  (>1: model reviews too often; <1: too rarely)
Uncertainty: n_draws draws of the causal bin estimates from N(beta, V) (cluster-robust covariance);
model curves are treated as fixed. Models without an item-specific path (DKT, SAKT, PFA, Elo) imply a
flat curve: no half-life, policy undefined. Outputs rq3_policy.csv, rq3_curves.csv.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_models import LAG_BINS, lag_bin  # noqa: E402

SEED = 42
ASYM = -1
FINE = False
MODELS_DIR = None
LAGS = [lab for _, _, lab in LAG_BINS]


def half_life(hours, effect):
    """First lag at which the normalized curve falls to 0.5 (log-linear interpolation); nan if no decay."""
    e = np.asarray(effect, float)
    top, bottom = e[0], e[ASYM]
    if not np.isfinite(top - bottom) or top - bottom <= 1e-4:
        return np.nan
    r = (e - bottom) / (top - bottom)
    lh = np.log(hours)
    for k in range(1, len(r)):
        if r[k] <= 0.5 < r[k - 1] or (r[k] <= 0.5 and k == 1):
            w = (r[k - 1] - 0.5) / (r[k - 1] - r[k]) if r[k - 1] != r[k] else 0
            return float(np.exp(lh[k - 1] + w * (lh[k] - lh[k - 1])))
    return np.nan


def retained(hours, effect, h):
    e = np.asarray(effect, float)
    r = (e - e[ASYM]) / (e[0] - e[ASYM])
    return float(np.interp(np.log(h), np.log(hours), r)) if np.isfinite(h) else np.nan


def main(rq1, rq2, n_draws=2000):
    rng = np.random.default_rng(SEED)
    # representative lag (geometric mean) of each bin among test re-exposures
    pr = pd.read_parquet(os.path.join(rq2, "rq2_test_predictions.parquet"), columns=["exposure", "log_gap_h"])
    pr = pr[pr.exposure > 1]
    hrs = np.expm1(pr.log_gap_h.values)
    lb = lag_bin(hrs)
    rep_h = np.array([np.exp(np.log(np.maximum(hrs[lb == b], 1e-3)).mean()) for b in LAGS])
    lags, src = LAGS, "test"
    if FINE:  # whole-period causal curve on the finer grid
        fh = pd.read_csv(os.path.join(rq1, "rq1_fine_lag_hours.csv"))
        lags, rep_h, src = fh.lag_bin.tolist(), fh.hours.values, "fine"
    ce = pd.read_csv(os.path.join(rq1, f"rq1_effects_{src}.csv"))
    ce = ce[(ce.spec == "C lag x prev") & ce.estimator.str.startswith("E1")].set_index("term")
    terms = pd.read_csv(os.path.join(rq1, f"rq1_cov_{src}_C_terms.csv")).iloc[:, 0].tolist()
    V = np.load(os.path.join(rq1, f"rq1_cov_{src}_C.npy"))

    mdir = MODELS_DIR or rq2
    curves = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(mdir, "rq2_lagshift_*.csv")))])
    iso = IsotonicRegression(increasing=False)
    rows, curve_rows = [], []
    for pv, lab in ((1, "after correct"), (0, "after error")):
        names = [f"{lab}, lag {b}" for b in lags]
        ix = [terms.index(n) for n in names]
        mu, Vg = ce.loc[names, "coef"].values, V[np.ix_(ix, ix)]
        draws = rng.multivariate_normal(mu, Vg, size=n_draws)

        def causal_curve(b):
            return iso.fit_transform(np.log(rep_h), b)
        c_hat = causal_curve(mu)
        H_c = half_life(rep_h, c_hat)
        H_c_draws = np.array([half_life(rep_h, causal_curve(d)) for d in draws])
        for h, b, e, ei in zip(rep_h, lags, mu, c_hat):
            curve_rows.append(dict(group=lab, source="causal", hours=h, effect=e, effect_isotonic=ei))
        rows.append(dict(group=lab, model="CAUSAL (reference)", half_life_h=H_c,
                         half_life_lo=np.nanpercentile(H_c_draws, 2.5), half_life_hi=np.nanpercentile(H_c_draws, 97.5),
                         retained_at_review=0.5, retained_lo=np.nan, retained_hi=np.nan,
                         review_rate_ratio=1.0, ratio_lo=np.nanpercentile(H_c / H_c_draws, 2.5) if np.isfinite(H_c) else np.nan,
                         ratio_hi=np.nanpercentile(H_c / H_c_draws, 97.5) if np.isfinite(H_c) else np.nan))
        for model, g in curves[curves.prev_y == pv].groupby("model", sort=False):
            g = g.sort_values("hours")
            # same anchoring as the causal curve: model curve read at the representative bin lags
            e_m = np.interp(np.log(rep_h), np.log(g.hours.values), g.implied.values)
            H_m = half_life(rep_h, e_m)
            for h, e in zip(g.hours, g.implied):
                curve_rows.append(dict(group=lab, source=model, hours=h, effect=e, effect_isotonic=np.nan))
            ret = np.array([retained(rep_h, causal_curve(d), H_m) for d in draws])
            ratio = H_c_draws / H_m if np.isfinite(H_m) else np.full(n_draws, np.nan)
            rows.append(dict(group=lab, model=model, half_life_h=H_m, half_life_lo=np.nan, half_life_hi=np.nan,
                             retained_at_review=retained(rep_h, c_hat, H_m),
                             retained_lo=np.nanpercentile(ret, 2.5) if np.isfinite(H_m) else np.nan,
                             retained_hi=np.nanpercentile(ret, 97.5) if np.isfinite(H_m) else np.nan,
                             review_rate_ratio=H_c / H_m if np.isfinite(H_m) else np.nan,
                             ratio_lo=np.nanpercentile(ratio, 2.5) if np.isfinite(H_m) else np.nan,
                             ratio_hi=np.nanpercentile(ratio, 97.5) if np.isfinite(H_m) else np.nan))
        for model in ("M7 DKT", "M7 SAKT", "M3 PFA", "M4 Elo (online)"):
            rows.append(dict(group=lab, model=model + " (no item path)", half_life_h=np.nan))
    res = pd.DataFrame(rows)
    sfx = ("" if ASYM == -1 else f"_asym{ASYM}") + ("_fine" if FINE else "")
    res.to_csv(os.path.join(mdir, f"rq3_policy{sfx}.csv"), index=False, float_format="%.4f")
    pd.DataFrame(curve_rows).to_csv(os.path.join(mdir, f"rq3_curves{sfx}.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 240)
    print("representative lag per bin (h):", np.round(rep_h, 1))
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    FINE = "--fine" in sys.argv
    if "--models-dir" in sys.argv:
        MODELS_DIR = sys.argv[sys.argv.index("--models-dir") + 1]
    if "--asym-bin" in sys.argv:
        ASYM = int(sys.argv[sys.argv.index("--asym-bin") + 1])
    nd = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 2000
    main(sys.argv[1], sys.argv[2], nd)
