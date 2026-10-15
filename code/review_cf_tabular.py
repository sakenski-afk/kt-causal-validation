"""
Reviewer checks for the tabular models on the test window F3 (2026-01..03):

    python review_cf_tabular.py <clean_dir> <out_dir>

(a) Temporal leakage in the online Elo features. In the pipeline the Elo states are updated row by row in the order
    (game_time, game_id, uid, pos); both players of an asynchronous duel answer the same three items, so the item rating
    can already contain the opponent's answer to the same item when the second player is predicted. Variant 'batched':
    all rows of a game are predicted with the state before the game and updated afterwards. GBM (with and without memory
    features) is retrained with each variant; test metrics and implied effects are compared.
(b) Counterfactual protocols for GBM:
      direct  only the item-memory features are reset to 'never seen' (as in the paper);
      full    in addition, the item's earlier encounters are removed from the KC and overall practice counts and the
              derived accuracies (Elo states and the recent-accuracy window are left unchanged: they cannot be rebuilt
              without re-running the whole stream).
(c) PFA: share of the KC practice count that the item's own earlier encounters represent (explains why deleting them
    barely moves PFA, which has no item-specific path).
Per-row implied effects on test re-exposures are written for the summary script.
"""
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402

END = "2026-04-01"
B.T_VAL, B.T_TEST = "2025-10-01", "2026-01-01"  # main split F3, as in rq2_models.py


def elo_batched(df, a=1.0, b=0.05):
    """Online Elo as B.elo, but all rows of one game are predicted with the state before that game."""
    u = pd.factorize(df.uid)[0]
    uk = pd.factorize(df.uid.astype(str) + "_" + df.cat_id.astype(str))[0]
    it = pd.factorize(df.question_id)[0]
    g = df.game_id.values
    th, thk, d = np.zeros(u.max() + 1), np.zeros(uk.max() + 1), np.zeros(it.max() + 1)
    nu, nuk, ni = np.zeros_like(th), np.zeros_like(thk), np.zeros_like(d)
    y = df.y.values.astype(float)
    p_out, th_out, thk_out, d_out, ni_out = (np.empty(len(df)) for _ in range(5))
    r, n = 0, len(df)
    while r < n:
        e_ = r
        while e_ < n and g[e_] == g[r]:
            e_ += 1
        for k in range(r, e_):  # predictions with the pre-game state
            i, j, q = u[k], uk[k], it[k]
            p_out[k] = 1.0 / (1.0 + np.exp(-(th[i] + thk[j] - d[q])))
            th_out[k], thk_out[k], d_out[k], ni_out[k] = th[i], thk[j], d[q], ni[q]
        for k in range(r, e_):  # updates after the game
            i, j, q = u[k], uk[k], it[k]
            e = y[k] - p_out[k]
            th[i] += a / (1 + b * nu[i]) * e
            thk[j] += a / (1 + b * nuk[j]) * e
            d[q] -= a / (1 + b * ni[q]) * e
            nu[i] += 1; nuk[j] += 1; ni[q] += 1
        r = e_
    return p_out, th_out, thk_out, d_out, ni_out


def main(clean, out):
    os.makedirs(out, exist_ok=True)
    df = B.load_features(clean)
    df = df[df.game_time < pd.Timestamp(END)].reset_index(drop=True)
    tr, va, te = ((df.split == s).values for s in ("train", "val", "test"))
    y = df.y.values
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(["uid", "question_id", "game_time", "game_id"])
    df["item_prior_s"] = df.groupby(["uid", "question_id"]).y.cumsum() - df.y
    df = df.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)
    df["kc_prior_acc"] = np.where(df.kc_prior_n > 0, df.kc_prior_s / df.kc_prior_n.clip(lower=1), np.nan)
    df["all_prior_acc"] = np.where(df.all_prior_n > 0, df.all_prior_s / df.all_prior_n.clip(lower=1), np.nan)
    seen = df.exposure.values > 1
    tsel = te & seen

    # (c) PFA: weight of the item's own encounters in the KC counts
    share = ((df.exposure - 1) / df.kc_prior_n.clip(lower=1))[tsel]
    pfa = pd.DataFrame([dict(n=int(tsel.sum()), mean_share=share.mean(), median_share=share.median(),
                             p90_share=share.quantile(0.9), median_kc_prior_n=df.kc_prior_n[tsel].median())])
    pfa.to_csv(os.path.join(out, "pfa_item_share_of_kc_counts.csv"), index=False, float_format="%.5f")
    print(pfa.to_string(index=False), flush=True)

    base = ["elo_theta", "elo_theta_kc", "elo_d", "item_n_seen", "kc_prior_n", "kc_prior_acc",
            "all_prior_n", "all_prior_acc", "recent_acc", "log_prior_games", "obs_share", "opp_is_bot",
            "history_complete", "hour", "cat_id"]
    memory = ["exposure", "prev_y", "log_gap_h"]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=B.SEED, verbose=-1)
    met, rows = [], df.loc[tsel, ["uid", "game_id", "question_id", "exposure", "prev_y", "log_gap_h"]].copy()
    for variant, fn in (("pipeline", B.elo), ("batched", elo_batched)):
        _, th, thk, d, ni = fn(df)
        df["elo_theta"], df["elo_theta_kc"], df["elo_d"], df["item_n_seen"] = th, thk, d, ni
        print("Elo", variant, "done", flush=True)
        for name, feats in (("M5 GBM (all features)", base + memory), ("M5a GBM without memory features", base)):
            m = lgb.train(params, lgb.Dataset(df.loc[tr, feats], y[tr], categorical_feature=["cat_id"]), 2000,
                          valid_sets=[lgb.Dataset(df.loc[va, feats], y[va], categorical_feature=["cat_id"])],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            p = m.predict(df[feats], num_iteration=m.best_iteration)
            met.append(dict(elo=variant, model=name, iters=m.best_iteration, **B.metrics(y[te].astype(float), p[te])))
            print(met[-1], flush=True)
            if "exposure" not in feats:
                continue
            Xd = df.loc[tsel, feats].copy()
            Xd["exposure"], Xd["prev_y"], Xd["log_gap_h"] = 1, -1, -1.0
            rows[f"direct | {variant}"] = p[tsel] - m.predict(Xd, num_iteration=m.best_iteration)
            if variant == "pipeline":
                Xf = Xd.copy()
                k = (df.exposure.values - 1)[tsel]
                s_ = df.item_prior_s.values[tsel]
                kn = np.maximum(df.kc_prior_n.values[tsel] - k, 0); ks = np.maximum(df.kc_prior_s.values[tsel] - s_, 0)
                an = np.maximum(df.all_prior_n.values[tsel] - k, 0); as_ = np.maximum(df.all_prior_s.values[tsel] - s_, 0)
                Xf["kc_prior_n"], Xf["all_prior_n"] = kn, an
                Xf["kc_prior_acc"] = np.where(kn > 0, ks / np.maximum(kn, 1), np.nan)
                Xf["all_prior_acc"] = np.where(an > 0, as_ / np.maximum(an, 1), np.nan)
                rows["full | pipeline"] = p[tsel] - m.predict(Xf, num_iteration=m.best_iteration)
    pd.DataFrame(met).to_csv(os.path.join(out, "elo_leakage_gbm_metrics.csv"), index=False, float_format="%.5f")
    rows.to_parquet(os.path.join(out, "cf_protocols_gbm_rows.parquet"), index=False)
    print(pd.DataFrame(met).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(rows[[c for c in rows.columns if "|" in c]].mean().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
