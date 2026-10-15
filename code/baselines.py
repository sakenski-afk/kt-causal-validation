"""
Baseline comparison for next-response correctness prediction (item level, round 5).

    python baselines.py <preprocessed_dir> <out_dir>

Temporal split (no shuffling):
    train       game_time <  2026-01-01
    validation  2026-01-01 <= game_time < 2026-04-01   (early stopping / hyper-parameters)
    test        game_time >= 2026-04-01
All features are computed from information available strictly before the response
(rounds 1-4 of the same game precede round 5 and may be used).

Models
    M0  Global mean
    M1  Item difficulty (smoothed train p-correct; category mean for unseen items)
    M2  IRT 1PL (Rasch): L2-regularised logistic regression on user + item indicators
    M3  PFA: item indicators + KC-specific success / failure counts
    M4  Elo (learner global + learner-KC + item), online, uncertainty function a/(1+b*n)
    M5  Gradient boosting (LightGBM) on all features incl. Elo state
    M5a Ablation: M5 without memory features (exposure, prev_y, gap)
Metrics: AUC, RMSE, log loss, ECE (10 equal-width bins); 95% CI by player-clustered bootstrap.
"""
import json
import os
import sys

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

T_VAL, T_TEST = "2026-01-01", "2026-04-01"
SEED = 42
N_BOOT = 200


def load_features(src):
    c = duckdb.connect()
    c.execute("SET memory_limit='5GB'")
    P = lambda n: f"'{os.path.join(src, n + '.parquet')}'"
    # causal cumulative counts on the full KC-level stream
    c.execute(f"""
        CREATE TABLE kc AS
        SELECT uid, game_id, round, pos, cat_id, y,
               kc_attempt - 1 AS kc_prior_n,
               coalesce(sum(y) OVER (PARTITION BY uid, cat_id ORDER BY kc_attempt
                                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS kc_prior_s,
               seq_idx - 1 AS all_prior_n,
               coalesce(sum(y) OVER (PARTITION BY uid ORDER BY seq_idx
                                     ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0) AS all_prior_s,
               coalesce(avg(y) OVER (PARTITION BY uid ORDER BY seq_idx
                                     ROWS BETWEEN 150 PRECEDING AND 1 PRECEDING), NULL) AS recent_acc
        FROM {P('kc_responses')}""")
    df = c.sql(f"""
        SELECT i.uid, i.question_id, i.cat_id, i.game_time, i.game_id, i.y::INT AS y,
               i.exposure, coalesce(i.prev_y, -1) AS prev_y,
               CASE WHEN i.gap_seconds IS NULL THEN -1 ELSE ln(1 + i.gap_seconds / 3600.0) END AS log_gap_h,
               i.history_complete, i.opp_is_bot::INT AS opp_is_bot, i.high_volume::INT AS high_volume,
               ln(1 + i.prior_games_all) AS log_prior_games,
               CASE WHEN i.prior_games_all > 0 THEN i.prior_games_observed / i.prior_games_all ELSE 1 END AS obs_share,
               k.kc_prior_n, k.kc_prior_s, k.all_prior_n, k.all_prior_s, k.recent_acc,
               hour(i.game_time) AS hour
        FROM {P('item_responses')} i
        JOIN kc k ON k.uid = i.uid AND k.game_id = i.game_id AND k.round = 5 AND k.pos = i.pos
        ORDER BY i.game_time, i.game_id, i.uid, i.pos""").df()
    df["split"] = np.where(df.game_time < pd.Timestamp(T_VAL), "train",
                           np.where(df.game_time < pd.Timestamp(T_TEST), "val", "test"))
    return df


def ece(y, p, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(y[m].mean() - p[m].mean())
    return tot / len(y)


def metrics(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AUC": roc_auc_score(y, p), "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
            "LogLoss": log_loss(y, p), "ECE": ece(y, p)}


def cluster_bootstrap(y, preds, groups, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(inv)
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    res = {m: [] for m in preds}
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        rows = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        for m, p in preds.items():
            res[m].append(roc_auc_score(y[rows], p[rows]))
    return {m: np.percentile(v, [2.5, 97.5]) for m, v in res.items()}, res


def item_difficulty(train, df, alpha=10):
    g = train.groupby("question_id").y.agg(["sum", "count"])
    cat = train.groupby("cat_id").y.mean()
    prior = df.cat_id.map(cat).fillna(train.y.mean())
    s = df.question_id.map(g["sum"]).fillna(0)
    n = df.question_id.map(g["count"]).fillna(0)
    return ((s + alpha * prior) / (n + alpha)).values


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def elo(df, a=1.0, b=0.05):
    """Online Elo: p = sigmoid(theta_u + theta_uk - d_i). Returns pre-update predictions and states."""
    u = pd.factorize(df.uid)[0]
    uk = pd.factorize(df.uid.astype(str) + "_" + df.cat_id.astype(str))[0]
    it = pd.factorize(df.question_id)[0]
    th, thk, d = np.zeros(u.max() + 1), np.zeros(uk.max() + 1), np.zeros(it.max() + 1)
    nu, nuk, ni = np.zeros_like(th), np.zeros_like(thk), np.zeros_like(d)
    y = df.y.values.astype(float)
    p_out, th_out, thk_out, d_out, ni_out = (np.empty(len(df)) for _ in range(5))
    for r in range(len(df)):
        i, j, q = u[r], uk[r], it[r]
        z = th[i] + thk[j] - d[q]
        p = 1.0 / (1.0 + np.exp(-z))
        p_out[r], th_out[r], thk_out[r], d_out[r], ni_out[r] = p, th[i], thk[j], d[q], ni[q]
        e = y[r] - p
        th[i] += a / (1 + b * nu[i]) * e
        thk[j] += a / (1 + b * nuk[j]) * e
        d[q] -= a / (1 + b * ni[q]) * e
        nu[i] += 1; nuk[j] += 1; ni[q] += 1
    return p_out, th_out, thk_out, d_out, ni_out


def main(src, out):
    os.makedirs(out, exist_ok=True)
    df = load_features(src)
    print(df.split.value_counts().to_dict(), flush=True)
    tr, va, te = ((df.split == s).values for s in ("train", "val", "test"))
    trv = tr | va
    y = df.y.values
    preds = {}

    # M0
    preds["M0 Global mean"] = np.full(len(df), y[trv].mean())

    # M1 (fit on train+val for test predictions)
    df["item_p"] = item_difficulty(df[trv], df)
    preds["M1 Item difficulty"] = df.item_p.values

    # M2 IRT 1PL
    uid_codes, uid_uniq = pd.factorize(df.uid)
    q_codes, q_uniq = pd.factorize(df.question_id)
    X = sparse.hstack([sparse.csr_matrix((np.ones(len(df)), (np.arange(len(df)), uid_codes)), shape=(len(df), len(uid_uniq))),
                       sparse.csr_matrix((np.ones(len(df)), (np.arange(len(df)), q_codes)), shape=(len(df), len(q_uniq)))]).tocsr()
    irt = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300)
    irt.fit(X[trv], y[trv])
    preds["M2 IRT-1PL"] = irt.predict_proba(X)[:, 1]
    print("IRT done", flush=True)

    # M3 PFA (KC-specific success/failure slopes + item easiness offset)
    kc_codes, kc_uniq = pd.factorize(df.cat_id)
    s_ = np.log1p(df.kc_prior_s.values)
    f_ = np.log1p(df.kc_prior_n.values - df.kc_prior_s.values)
    rows = np.arange(len(df))
    K = len(kc_uniq)
    Xp = sparse.hstack([
        sparse.csr_matrix((np.ones(len(df)), (rows, q_codes)), shape=(len(df), len(q_uniq))),
        sparse.csr_matrix((s_, (rows, kc_codes)), shape=(len(df), K)),
        sparse.csr_matrix((f_, (rows, kc_codes)), shape=(len(df), K)),
        sparse.csr_matrix((np.ones(len(df)), (rows, kc_codes)), shape=(len(df), K))]).tocsr()
    pfa = LogisticRegression(C=1.0, max_iter=500)
    pfa.fit(Xp[trv], y[trv])
    preds["M3 PFA"] = pfa.predict_proba(Xp)[:, 1]
    print("PFA done", flush=True)

    # M4 Elo (online over the whole chronological stream)
    p_elo, th, thk, d, ni = elo(df)
    preds["M4 Elo (online)"] = p_elo
    df["elo_theta"], df["elo_theta_kc"], df["elo_d"], df["item_n_seen"] = th, thk, d, ni
    print("Elo done", flush=True)

    # M5 LightGBM
    df["kc_prior_acc"] = np.where(df.kc_prior_n > 0, df.kc_prior_s / df.kc_prior_n.clip(lower=1), np.nan)
    df["all_prior_acc"] = np.where(df.all_prior_n > 0, df.all_prior_s / df.all_prior_n.clip(lower=1), np.nan)
    # item difficulty enters through the causal Elo state (elo_d); no target-encoded item means
    base = ["elo_theta", "elo_theta_kc", "elo_d", "item_n_seen", "kc_prior_n", "kc_prior_acc",
            "all_prior_n", "all_prior_acc", "recent_acc", "log_prior_games", "obs_share", "opp_is_bot",
            "history_complete", "hour", "cat_id"]
    memory = ["exposure", "prev_y", "log_gap_h"]
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, seed=SEED, verbose=-1)
    importance = {}
    for name, feats in (("M5 GBM (all features)", base + memory), ("M5a GBM without memory features", base)):
        dtr = lgb.Dataset(df.loc[tr, feats], y[tr], categorical_feature=["cat_id"])
        dva = lgb.Dataset(df.loc[va, feats], y[va], categorical_feature=["cat_id"])
        m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        preds[name] = m.predict(df[feats], num_iteration=m.best_iteration)
        importance[name] = dict(zip(feats, m.feature_importance("gain").round(0).tolist()))
        print(name, "iters", m.best_iteration, flush=True)

    # ---- evaluation on test ----
    rows_out = []
    yt = y[te]
    test_preds = {k: v[te] for k, v in preds.items()}
    ci, boot = cluster_bootstrap(yt, test_preds, df.uid.values[te])
    for subset, mask in (("test, all players", np.ones(te.sum(), bool)),
                         ("test, cohort 2025+", df.history_complete.values[te] == 1),
                         ("test, first exposure", df.exposure.values[te] == 1),
                         ("test, re-exposure", df.exposure.values[te] > 1)):
        for k, p in test_preds.items():
            r = {"subset": subset, "model": k, "n": int(mask.sum()), **metrics(yt[mask], p[mask])}
            if subset == "test, all players":
                r["AUC_CI_low"], r["AUC_CI_high"] = ci[k]
            rows_out.append(r)
    res = pd.DataFrame(rows_out)
    res.to_csv(os.path.join(out, "baseline_results.csv"), index=False, float_format="%.4f")
    # paired bootstrap: GBM vs best non-GBM and GBM vs ablation
    b = {k: np.array(v) for k, v in boot.items()}
    paired = {
        "M5 - M4 AUC diff CI": np.percentile(b["M5 GBM (all features)"] - b["M4 Elo (online)"], [2.5, 97.5]).tolist(),
        "M5 - M2 AUC diff CI": np.percentile(b["M5 GBM (all features)"] - b["M2 IRT-1PL"], [2.5, 97.5]).tolist(),
        "M5 - M5a AUC diff CI": np.percentile(b["M5 GBM (all features)"] - b["M5a GBM without memory features"], [2.5, 97.5]).tolist(),
    }
    with open(os.path.join(out, "baseline_extra.json"), "w") as f:
        json.dump({"split_sizes": df.split.value_counts().to_dict(), "paired_bootstrap": paired,
                   "gbm_gain_importance": importance}, f, indent=2, ensure_ascii=False)
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
