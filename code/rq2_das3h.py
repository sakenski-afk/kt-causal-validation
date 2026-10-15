"""
RQ2 stage 2a: DAS3H (Choffin et al., 2019) with and without item-level time windows.

    python rq2_das3h.py <clean_dir> <out_dir> [max_iter]   (default 2000; the first run used 500)

logit p = a_u - d_i + b_k + sum_w [ t_{k,w}^c log1p(correct_{u,k,w}) + t_{k,w}^a log1p(attempts_{u,k,w}) ]
  k = category (KC), windows w = 1 h, 1 d, 7 d, 30 d, all time; counts over the KC-level stream
  (rounds 1-5) strictly before the current game.
DAS3H+item adds the same five windows for the learner's earlier round-5 encounters with this item,
with parameters shared across items (explicit item memory with forgetting).
Fit: L2 logistic regression (C = 1, lbfgs) on train+val, as M2/M3 in the pipeline; split as in
rq2_models.py (train < 2025-10-01 <= val < 2026-01-01 <= test < 2026-04-01; data before 2026-04-01).
--fold T_VAL T_TEST END: rolling-origin fold (train < T_VAL <= val < T_TEST <= test < END); outputs go to
<out_dir>/fold_<T_TEST yyyy-mm>. Default = the main split 2025-10-01 / 2026-01-01 / 2026-04-01.
Model-implied effect of a re-exposure: p(actual) - p(the item's earlier encounters removed), i.e. item
windows set to 0 and the item's encounters subtracted from the KC windows. Output format = stage 1.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402
from rq2_models import GRID_H, LAG_BINS, lag_bin  # noqa: E402

END, T_VAL, T_TEST = "2026-04-01", "2025-10-01", "2026-01-01"
WINDOWS = [("1h", "1 hour"), ("1d", "1 day"), ("7d", "7 days"), ("30d", "30 days")]


def main(clean, out, max_iter=2000):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    c.execute(f"""CREATE TABLE kcg AS SELECT uid, cat_id, game_id, min(game_time) t, count(*) a, sum(y::INT) s
                  FROM {P('kc_responses')} WHERE game_time < TIMESTAMP '{END}' GROUP BY ALL""")
    wk = []
    for lab, iv in WINDOWS:
        wk.append(f"sum(a) OVER (PARTITION BY uid, cat_id ORDER BY t RANGE BETWEEN INTERVAL '{iv}' PRECEDING AND CURRENT ROW) - a AS ka_{lab}")
        wk.append(f"sum(s) OVER (PARTITION BY uid, cat_id ORDER BY t RANGE BETWEEN INTERVAL '{iv}' PRECEDING AND CURRENT ROW) - s AS ks_{lab}")
    wk.append("sum(a) OVER (PARTITION BY uid, cat_id ORDER BY t ROWS UNBOUNDED PRECEDING) - a AS ka_all")
    wk.append("sum(s) OVER (PARTITION BY uid, cat_id ORDER BY t ROWS UNBOUNDED PRECEDING) - s AS ks_all")
    c.execute(f"CREATE TABLE kcw AS SELECT uid, cat_id, game_id, {', '.join(wk)} FROM kcg")
    iw = []
    for lab, iv in WINDOWS:
        iw.append(f"count(*) OVER (PARTITION BY uid, question_id ORDER BY game_time RANGE BETWEEN INTERVAL '{iv}' PRECEDING AND CURRENT ROW) - 1 AS ia_{lab}")
        iw.append(f"sum(y::INT) OVER (PARTITION BY uid, question_id ORDER BY game_time RANGE BETWEEN INTERVAL '{iv}' PRECEDING AND CURRENT ROW) - y::INT AS is_{lab}")
    iw.append("exposure - 1 AS ia_all")
    iw.append("sum(y::INT) OVER (PARTITION BY uid, question_id ORDER BY game_time, game_id ROWS UNBOUNDED PRECEDING) - y::INT AS is_all")
    df = c.sql(f"""
        WITH it AS (SELECT uid, game_id, pos, game_time, cat_id, question_id, y::INT AS y, exposure, prev_y, gap_seconds,
                           {', '.join(iw)}
                    FROM {P('item_responses')} WHERE game_time < TIMESTAMP '{END}')
        SELECT it.*, kcw.* EXCLUDE (uid, cat_id, game_id) FROM it JOIN kcw USING (uid, cat_id, game_id)
        ORDER BY game_time, game_id, uid, pos""").df()
    df["split"] = np.where(df.game_time < pd.Timestamp(T_VAL), "train", np.where(df.game_time < pd.Timestamp(T_TEST), "val", "test"))
    print(df.split.value_counts().to_dict(), flush=True)
    labs = [l for l, _ in WINDOWS] + ["all"]
    for l in labs:  # numerical guard: counts are non-negative
        for p in ("ka", "ks", "ia", "is"):
            df[f"{p}_{l}"] = df[f"{p}_{l}"].clip(lower=0)

    n = len(df); rows = np.arange(n)
    u_codes, u_uniq = pd.factorize(df.uid)
    q_codes, q_uniq = pd.factorize(df.question_id)
    k_codes, k_uniq = pd.factorize(df.cat_id); K = len(k_uniq)

    def design(kc_cnt, item_cnt, with_item):
        blocks = [sparse.csr_matrix((np.ones(n), (rows, u_codes)), shape=(n, len(u_uniq))),
                  sparse.csr_matrix((np.ones(n), (rows, q_codes)), shape=(n, len(q_uniq))),
                  sparse.csr_matrix((np.ones(n), (rows, k_codes)), shape=(n, K))]
        for l in labs:  # per-KC window parameters
            for p in ("ka", "ks"):
                blocks.append(sparse.csr_matrix((np.log1p(kc_cnt[f"{p}_{l}"]), (rows, k_codes)), shape=(n, K)))
        if with_item:  # shared item-window parameters
            blocks.append(sparse.csr_matrix(np.column_stack([np.log1p(item_cnt[f"{p}_{l}"]) for l in labs for p in ("ia", "is")])))
        return sparse.hstack(blocks).tocsr()

    kc_act = {f"{p}_{l}": df[f"{p}_{l}"].values.astype(float) for l in labs for p in ("ka", "ks")}
    it_act = {f"{p}_{l}": df[f"{p}_{l}"].values.astype(float) for l in labs for p in ("ia", "is")}
    # counterfactual: the item's earlier encounters removed from KC windows; item windows zero
    kc_cf = {f"ka_{l}": np.maximum(kc_act[f"ka_{l}"] - it_act[f"ia_{l}"], 0) for l in labs}
    kc_cf.update({f"ks_{l}": np.maximum(kc_act[f"ks_{l}"] - it_act[f"is_{l}"], 0) for l in labs})
    it_cf = {k: np.zeros(n) for k in it_act}

    y = df.y.values
    te = (df.split == "test").values; trv = ~te
    seen = df.exposure.values > 1
    hours = df.gap_seconds.values / 3600.0
    lb = lag_bin(np.nan_to_num(hours, nan=-1))
    prev = df.prev_y.fillna(-1).values
    met, imp_rows = [], []
    rows_out = df.loc[te, ["uid", "game_id", "question_id", "y", "exposure"]].copy()
    rows_out["lag_bin"], rows_out["prev_y"] = lb[te], prev[te]
    for name, with_item in (("M6 DAS3H", False), ("M6b DAS3H + item windows", True)):
        X = design(kc_act, it_act, with_item)
        m = LogisticRegression(C=1.0, solver="lbfgs", max_iter=max_iter).fit(X[trv], y[trv])
        p = m.predict_proba(X)[:, 1]
        pcf = m.predict_proba(design(kc_cf, it_cf, with_item))[:, 1]
        met.append(dict(model=name, n=int(te.sum()), **B.metrics(y[te], p[te])))
        imp = p - pcf
        rows_out["p | " + name], rows_out["implied | " + name] = p[te], imp[te]
        if with_item:  # RQ3: implied effect of ONE earlier encounter at lag h (item windows), by its outcome
            curve_rows = []
            for pv in (1, 0):
                sel = te & seen & (prev == pv)
                idx = np.flatnonzero(sel)
                kc_q = {k: v[idx] for k, v in kc_cf.items()}
                n_q = len(idx)
                def design_q(item_cnt):
                    rows_q = np.arange(n_q)
                    blocks = [sparse.csr_matrix((np.ones(n_q), (rows_q, u_codes[idx])), shape=(n_q, len(u_uniq))),
                              sparse.csr_matrix((np.ones(n_q), (rows_q, q_codes[idx])), shape=(n_q, len(q_uniq))),
                              sparse.csr_matrix((np.ones(n_q), (rows_q, k_codes[idx])), shape=(n_q, K))]
                    for l in labs:
                        for pp in ("ka", "ks"):
                            blocks.append(sparse.csr_matrix((np.log1p(kc_q[f"{pp}_{l}"]), (rows_q, k_codes[idx])), shape=(n_q, K)))
                    blocks.append(sparse.csr_matrix(np.column_stack([np.log1p(item_cnt[f"{pp}_{l}"]) for l in labs for pp in ("ia", "is")])))
                    return sparse.hstack(blocks).tocsr()
                p0 = m.predict_proba(design_q({k: np.zeros(n_q) for k in it_act}))[:, 1]
                win_h = {"1h": 1, "1d": 24, "7d": 168, "30d": 720, "all": np.inf}
                for h in GRID_H:
                    cnt = {}
                    for l in labs:
                        inside = float(h < win_h[l])
                        cnt[f"ia_{l}"] = np.full(n_q, inside); cnt[f"is_{l}"] = np.full(n_q, inside * pv)
                    ph = m.predict_proba(design_q(cnt))[:, 1]
                    curve_rows.append(dict(model=name, prev_y=pv, hours=h, n=n_q, implied=float((ph - p0).mean())))
            pd.DataFrame(curve_rows).to_csv(os.path.join(out, "rq2_lagshift_M6b.csv"), index=False, float_format="%.5f")
        tsel = te & seen
        for lab, msk in [("all", tsel)] + [(f"lag {b}", tsel & (lb == b)) for _, _, b in LAG_BINS] + \
                        [(f"after correct, lag {b}", tsel & (lb == b) & (prev == 1)) for _, _, b in LAG_BINS] + \
                        [(f"after error, lag {b}", tsel & (lb == b) & (prev == 0)) for _, _, b in LAG_BINS]:
            imp_rows.append(dict(model=name, subset=lab, n=int(msk.sum()),
                                 implied_effect=float(imp[msk].mean()) if msk.any() else np.nan, item_specific_path=True))
        print(name, met[-1], "implied all:", round(imp[tsel].mean(), 4), "iters", m.n_iter_, flush=True)
    rows_out.to_parquet(os.path.join(out, "rq2_rows_das3h.parquet"), index=False)
    pd.DataFrame(met).to_csv(os.path.join(out, "rq2_metrics_test_das3h.csv"), index=False, float_format="%.4f")
    pd.DataFrame(imp_rows).to_csv(os.path.join(out, "rq2_implied_effects_das3h.csv"), index=False, float_format="%.5f")


from rq2_models import parse_fold  # noqa: E402

if __name__ == "__main__":
    tv, tt, en, fold_out = parse_fold(sys.argv, sys.argv[2])
    if tv:
        T_VAL, T_TEST, END = tv, tt, en
    it = int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3].isdigit() else 2000
    main(sys.argv[1], fold_out, it)
