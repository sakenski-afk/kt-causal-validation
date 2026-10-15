"""
RQ2 stage 2b, step 1: token stream for the deep KT models -> cache file (no torch in this process).

    python rq2_deep_data.py <clean_dir> <out_dir>

DuckDB result conversion and PyTorch must not run in the same Windows process (fatal GIL error on
.df() after torch is imported), so the token stream is prepared here and saved to
<out_dir>/deep_tokens.npz; rq2_deep.py only loads that file.
--fold T_VAL T_TEST END: rolling-origin fold (train < T_VAL <= val < T_TEST <= test < END); outputs go to
<out_dir>/fold_<T_TEST yyyy-mm>. Default = the main split 2025-10-01 / 2026-01-01 / 2026-04-01. See rq2_deep.py for the design.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

END, T_VAL, T_TEST = "2026-04-01", "2025-10-01", "2026-01-01"


def load_tokens(clean):
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    # round-5 responses with item ids come from item_responses; every other KC-level response
    # (rounds 1-4, and round-5 responses without a stored item id) from kc_responses (anti join)
    c.execute(f"""CREATE TABLE i AS SELECT uid, game_id, game_time, round, pos, cat_id, y::INT AS y, question_id,
                  exposure, prev_y, gap_seconds FROM {P('item_responses')} WHERE game_time < TIMESTAMP '{END}'""")
    c.execute(f"""CREATE TABLE k AS SELECT uid, game_id, game_time, round, pos, cat_id, y::INT AS y
                  FROM {P('kc_responses')} WHERE game_time < TIMESTAMP '{END}'""")
    c.execute("""CREATE TABLE k_rest AS SELECT k.* FROM k ANTI JOIN i USING (uid, game_id, round, pos)""")
    c.execute("CREATE TABLE cats AS SELECT cat_id, (row_number() OVER (ORDER BY cat_id) - 1) AS cat_ix FROM (SELECT DISTINCT cat_id FROM k)")
    c.execute("CREATE TABLE qs AS SELECT question_id, row_number() OVER (ORDER BY question_id) AS q_ix FROM (SELECT DISTINCT question_id FROM i)")
    df = c.sql("""
        WITH u AS (SELECT uid, game_id, game_time, round, pos, cat_id, y, question_id, exposure, prev_y, gap_seconds FROM i
                   UNION ALL
                   SELECT uid, game_id, game_time, round, pos, cat_id, y, NULL, NULL, NULL, NULL FROM k_rest)
        SELECT u.*, cats.cat_ix, qs.q_ix FROM u JOIN cats USING (cat_id) LEFT JOIN qs USING (question_id)
        ORDER BY u.uid, u.game_time, u.game_id, u.round, u.pos""").df()
    df.attrs["n_q"] = c.execute("SELECT count(*) FROM qs").fetchone()[0]
    return df


def build(df):
    nq = df.attrs["n_q"]
    n_cat = int(df.cat_ix.max()) + 1
    cat = df.cat_ix.values.astype(np.int64)
    is5 = df.question_id.notna().values
    item = np.where(is5, df.q_ix.fillna(0).values.astype(np.int64), nq + 1 + cat)  # 0 = padding
    t = df.game_time.values.astype("datetime64[s]").astype(np.int64)
    uid = df.uid.values
    # time since previous token of the learner (hours)
    dt_prev = np.zeros(len(df))
    same = np.r_[False, uid[1:] == uid[:-1]]
    dt_prev[same] = (t[1:] - t[:-1])[same[1:]] / 3600.0
    gap_h = np.where(is5 & (df.exposure.fillna(1).values > 1), df.gap_seconds.fillna(0).values / 3600.0, 0.0)
    n_prev = np.where(is5, df.exposure.fillna(1).values - 1, 0).astype(float)
    feats = np.column_stack([np.log1p(np.maximum(dt_prev, 0)), np.log1p(gap_h), np.log1p(n_prev),
                             (n_prev > 0).astype(float)]).astype(np.float32)
    split = np.where(df.game_time.values < np.datetime64(T_VAL), 0, np.where(df.game_time.values < np.datetime64(T_TEST), 1, 2))
    starts = np.flatnonzero(~same)
    ends = np.r_[starts[1:], len(df)]
    return dict(item=item.astype(np.int64), cat=cat, y=df.y.values.astype(np.int64), is5=is5, feats=feats,
                split=split, starts=starts, ends=ends, n_item_tok=nq + 1 + n_cat, n_cat=n_cat,
                uid=uid, game_id=df.game_id.values.astype(np.int64), question_id=df.question_id.fillna(-1).values.astype(np.int64), gap_h=gap_h, prev_y=df.prev_y.fillna(-1).values, exposure=df.exposure.fillna(1).values)



def main(clean, out):
    os.makedirs(out, exist_ok=True)
    data = build(load_tokens(clean))
    np.savez(os.path.join(out, "deep_tokens.npz"), **{k: np.asarray(v) for k, v in data.items()})
    print("tokens:", len(data["y"]), "round-5 tokens:", int(data["is5"].sum()), "learners:", len(data["starts"]), flush=True)


if __name__ == "__main__":
    if "--fold" in sys.argv:
        k = sys.argv.index("--fold")
        T_VAL, T_TEST, END = sys.argv[k + 1:k + 4]
        out_dir = os.path.join(sys.argv[2], "fold_" + T_TEST[:7])
    else:
        out_dir = sys.argv[2]
    main(sys.argv[1], out_dir)
