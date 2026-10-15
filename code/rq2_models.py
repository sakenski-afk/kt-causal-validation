"""
RQ2 stage 1: refit the pipeline models on the clean period and extract model-implied effects.

    python rq2_models.py <clean_dir> <out_dir> [--pfa-iter N]   (pipeline value 500; sensitivity 2000)

Period: games before 2026-04-01 (data-owner decision). Temporal split inside it:
    train < 2025-10-01 <= val < 2026-01-01 <= test < 2026-04-01
Models, features, hyper-parameters and seed are those of baselines.py (imported, not changed);
only the split dates differ. M0-M5a as in the pipeline.

Model-implied effects on test re-exposures (the same counterfactual query for every model):
    implied_seen = p(actual history) - p(history in which the learner has not met this item before)
  - M0, M1, M2 (static, no learner history of the item): implied effect is 0 by construction.
  - M3 PFA: the earlier encounter(s) with the item enter only through the KC success/failure counts;
    counterfactual removes them (n -= exposure-1, s -= earlier successes on this item).
  - M4 Elo: learner states carry no item-specific memory; the earlier encounter moved the global and
    KC ratings. Counterfactual: remove that update approximately is not attempted here -> reported as
    'no item-specific path' (effect through the learner rating is common to all items in the game and
    is absorbed by the learner-game contrast of the causal estimand).
  - M5 GBM: memory features exposure, prev_y, log_gap_h set to 'never seen' (1, -1, -1).
  - M5a GBM without memory features: no item-specific path -> 0.
Implied effects are summarised exactly like the causal estimands: overall mean over test re-exposures,
by actual lag bin, and by previous outcome x lag bin. Also a lag-shift query for M5 (its internal
forgetting curve): log_gap_h set to fixed values for every test re-exposure.
--fold T_VAL T_TEST END: rolling-origin fold (train < T_VAL <= val < T_TEST <= test < END); outputs go to
<out_dir>/fold_<T_TEST yyyy-mm>. Default = the main split 2025-10-01 / 2026-01-01 / 2026-04-01.
Writes metrics, implied-effect tables and test predictions (no learner-level export beyond ids in a
local parquet used by later stages).
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402

END = "2026-04-01"
B.T_VAL, B.T_TEST = "2025-10-01", "2026-01-01"
LAG_BINS = [(0, 1, "<1h"), (1, 6, "1-6h"), (6, 24, "6-24h"), (24, 72, "1-3d"), (72, 168, "3-7d"),
            (168, 336, "7-14d"), (336, 720, "14-30d"), (720, 2160, "30-90d"), (2160, 10**9, ">90d")]
SHIFT_H = [0.5, 3, 12, 48, 120, 240, 528, 1440, 4320]
# fine grid (hours) for the model-implied forgetting curves used by RQ3
GRID_H = [0.25, 0.5, 1, 2, 4, 8, 12, 24, 48, 72, 120, 168, 240, 336, 504, 720, 1080, 1440, 2160, 3000, 4320]


def lag_bin(hours):
    out = np.full(len(hours), None, dtype=object)
    for a, b, lab in LAG_BINS:
        out[(hours >= a) & (hours < b)] = lab
    return out


def main(src, out, pfa_iter=500):
    os.makedirs(out, exist_ok=True)
    df = B.load_features(src)
    df = df[df.game_time < pd.Timestamp(END)].reset_index(drop=True)
    print(df.split.value_counts().to_dict(), flush=True)
    tr, va, te = ((df.split == s).values for s in ("train", "val", "test"))
    trv = tr | va
    y = df.y.values
    preds, cf = {}, {}

    # earlier successes on this very item (for the PFA counterfactual): cumulative y over previous encounters
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(["uid", "question_id", "game_time", "game_id"])
    df["item_prior_s"] = df.groupby(["uid", "question_id"]).y.cumsum() - df.y
    df = df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)

    preds["M0 Global mean"] = np.full(len(df), y[trv].mean())
    df["item_p"] = B.item_difficulty(df[trv], df)
    preds["M1 Item difficulty"] = df.item_p.values

    uid_codes, uid_uniq = pd.factorize(df.uid)
    q_codes, q_uniq = pd.factorize(df.question_id)
    n = len(df); rows = np.arange(n)
    X = sparse.hstack([sparse.csr_matrix((np.ones(n), (rows, uid_codes)), shape=(n, len(uid_uniq))),
                       sparse.csr_matrix((np.ones(n), (rows, q_codes)), shape=(n, len(q_uniq)))]).tocsr()
    irt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300).fit(X[trv], y[trv])
    preds["M2 IRT-1PL"] = irt.predict_proba(X)[:, 1]
    print("IRT done", flush=True)

    kc_codes, kc_uniq = pd.factorize(df.cat_id); K = len(kc_uniq)

    def pfa_X(kc_n, kc_s):
        s_ = np.log1p(kc_s); f_ = np.log1p(kc_n - kc_s)
        return sparse.hstack([sparse.csr_matrix((np.ones(n), (rows, q_codes)), shape=(n, len(q_uniq))),
                              sparse.csr_matrix((s_, (rows, kc_codes)), shape=(n, K)),
                              sparse.csr_matrix((f_, (rows, kc_codes)), shape=(n, K)),
                              sparse.csr_matrix((np.ones(n), (rows, kc_codes)), shape=(n, K))]).tocsr()
    Xp = pfa_X(df.kc_prior_n.values, df.kc_prior_s.values)
    pfa = LogisticRegression(C=1.0, max_iter=pfa_iter).fit(Xp[trv], y[trv])
    print('PFA iterations', pfa.n_iter_, flush=True)
    preds["M3 PFA"] = pfa.predict_proba(Xp)[:, 1]
    prior_enc = (df.exposure.values - 1)
    Xp_cf = pfa_X(np.maximum(df.kc_prior_n.values - prior_enc, 0),
                  np.maximum(df.kc_prior_s.values - df.item_prior_s.values, 0))
    cf["M3 PFA"] = pfa.predict_proba(Xp_cf)[:, 1]
    print("PFA done", flush=True)

    p_elo, th, thk, d, ni = B.elo(df)
    preds["M4 Elo (online)"] = p_elo
    df["elo_theta"], df["elo_theta_kc"], df["elo_d"], df["item_n_seen"] = th, thk, d, ni
    print("Elo done", flush=True)

    df["kc_prior_acc"] = np.where(df.kc_prior_n > 0, df.kc_prior_s / df.kc_prior_n.clip(lower=1), np.nan)
    df["all_prior_acc"] = np.where(df.all_prior_n > 0, df.all_prior_s / df.all_prior_n.clip(lower=1), np.nan)
    base = ["elo_theta", "elo_theta_kc", "elo_d", "item_n_seen", "kc_prior_n", "kc_prior_acc",
            "all_prior_n", "all_prior_acc", "recent_acc", "log_prior_games", "obs_share", "opp_is_bot",
            "history_complete", "hour", "cat_id"]
    memory = ["exposure", "prev_y", "log_gap_h"]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=B.SEED, verbose=-1)
    seen = df.exposure.values > 1
    shift_rows = []
    for name, feats in (("M5 GBM (all features)", base + memory), ("M5a GBM without memory features", base)):
        dtr = lgb.Dataset(df.loc[tr, feats], y[tr], categorical_feature=["cat_id"])
        dva = lgb.Dataset(df.loc[va, feats], y[va], categorical_feature=["cat_id"])
        m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds[name] = m.predict(df[feats], num_iteration=m.best_iteration)
        print(name, "iters", m.best_iteration, flush=True)
        if "exposure" in feats:
            Xc = df[feats].copy()
            Xc["exposure"], Xc["prev_y"], Xc["log_gap_h"] = 1, -1, -1.0
            cf[name] = m.predict(Xc, num_iteration=m.best_iteration)
            tsel = te & seen
            Xs = df.loc[tsel, feats].copy()
            for h in SHIFT_H:
                Xs["log_gap_h"] = np.log1p(h)
                shift_rows.append(dict(model=name, hours=h, mean_p=m.predict(Xs, num_iteration=m.best_iteration).mean()))
            Xs["exposure"], Xs["prev_y"], Xs["log_gap_h"] = 1, -1, -1.0
            p_unseen = m.predict(Xs, num_iteration=m.best_iteration).mean()
            for r in shift_rows:
                if r["model"] == name:
                    r["implied_effect_vs_unseen"] = r["mean_p"] - p_unseen
            # RQ3: implied effect of ONE earlier encounter at lag h, by outcome of that encounter
            # (test re-exposures with that previous outcome; memory features changed, rest kept)
            curve_rows = []
            for pv in (1, 0):
                Xq = df.loc[tsel & (df.prev_y.values == pv), feats].copy()
                Xu = Xq.copy(); Xu["exposure"], Xu["prev_y"], Xu["log_gap_h"] = 1, -1, -1.0
                pu = m.predict(Xu, num_iteration=m.best_iteration)
                Xq["exposure"] = 2  # exactly one earlier encounter, as in the DAS3H and DKT-time queries
                for h in GRID_H:
                    Xq["log_gap_h"] = np.log1p(h)
                    curve_rows.append(dict(model=name, prev_y=pv, hours=h, n=len(Xq),
                                           implied=float((m.predict(Xq, num_iteration=m.best_iteration) - pu).mean())))
            pd.DataFrame(curve_rows).to_csv(os.path.join(out, "rq2_lagshift_M5.csv"), index=False, float_format="%.5f")

    # ---- metrics on test (clean period) ----
    yt = y[te]
    met = [dict(model=k, n=int(te.sum()), **B.metrics(yt, v[te])) for k, v in preds.items()]
    pd.DataFrame(met).to_csv(os.path.join(out, "rq2_metrics_test.csv"), index=False, float_format="%.4f")

    # ---- implied effects on test re-exposures ----
    tsel = te & seen
    hours = np.expm1(df.log_gap_h.values)
    lb = lag_bin(hours)
    prev = df.prev_y.values
    rows_out = []
    for k, p in preds.items():
        imp = (p - cf[k]) if k in cf else np.zeros(len(df))
        for lab, m_ in [("all", tsel)] + [(f"lag {b}", tsel & (lb == b)) for _, _, b in LAG_BINS] + \
                       [(f"after correct, lag {b}", tsel & (lb == b) & (prev == 1)) for _, _, b in LAG_BINS] + \
                       [(f"after error, lag {b}", tsel & (lb == b) & (prev == 0)) for _, _, b in LAG_BINS]:
            rows_out.append(dict(model=k, subset=lab, n=int(m_.sum()),
                                 implied_effect=float(imp[m_].mean()) if m_.any() else np.nan,
                                 item_specific_path=k in cf))
    imp_df = pd.DataFrame(rows_out)
    imp_df.to_csv(os.path.join(out, "rq2_implied_effects.csv"), index=False, float_format="%.5f")
    pd.DataFrame(shift_rows).to_csv(os.path.join(out, "rq2_gbm_lag_shift.csv"), index=False, float_format="%.5f")

    keep = ["uid", "game_id", "question_id", "cat_id", "game_time", "split", "y", "exposure", "prev_y", "log_gap_h"]
    pr = df.loc[te, keep].copy()
    for k, v in preds.items():
        pr[k] = v[te]
        if k in cf:
            pr[k + " | cf unseen"] = cf[k][te]
    pr.to_parquet(os.path.join(out, "rq2_test_predictions.parquet"), index=False)

    pd.set_option("display.width", 220)
    print(pd.DataFrame(met).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(imp_df[imp_df.subset.isin(["all"] + [f"lag {b}" for _, _, b in LAG_BINS])]
          .pivot(index="subset", columns="model", values="implied_effect").to_string(float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(shift_rows).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def parse_fold(argv, out):
    """--fold T_VAL T_TEST END -> (t_val, t_test, end, out_dir)"""
    if "--fold" in argv:
        k = argv.index("--fold")
        t_val, t_test, end = argv[k + 1:k + 4]
        return t_val, t_test, end, os.path.join(out, "fold_" + t_test[:7])
    return None, None, None, out


if __name__ == "__main__":
    tv, tt, en, fold_out = parse_fold(sys.argv, sys.argv[2])
    if tv:
        B.T_VAL, B.T_TEST, END = tv, tt, en
    it = int(sys.argv[sys.argv.index("--pfa-iter") + 1]) if "--pfa-iter" in sys.argv else 500
    main(sys.argv[1], fold_out if it == 500 else os.path.join(fold_out, f"pfa_iter_{it}"), it)
