"""
Is the round-5 item draw independent of the learner? (design check for causal analyses)

    python randomization_check.py <clean_dir> <export_dir> <out_dir> [sample_frac]

H0: within a category x item-language pool and calendar window, the system draws the item without regard to the
learner's state. Under H0 the drawn item is a draw from the window's item distribution, so for
every learner-specific property X(u, i) of an item (seen before, last answer wrong, time since last encounter in bins)

    E[X(u, drawn)] = sum_i f_i X(u, i),   f_i = share of draws of item i in the window to OTHER learners

Draws are counted per game (both players receive the same items). All games of learner u are left out
of f (leave-learner-out): an item u has seen was by construction drawn to u, so counting u's own draws
would inflate the expected share of u's recently seen items and create spurious "suppression".
Non-uniform item popularity is allowed; only dependence on the learner is tested.
Reported: observed - expected, 95% CI by learner-clustered bootstrap, by window (week / month /
quarter), period, learner side (initiator / opponent) and learner strata; plus a regression of the
difficulty of the drawn item minus the window mean on prior learner accuracy.
Only aggregates are written.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 500


def main(clean, export, out, frac=0.1):
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "_duckdb_tmp")
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute(f"SET temp_directory='{tmp}'")
    c.execute("SET preserve_insertion_order=false")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"

    # pool key = category x item language (Kazakh / Russian items form separate pools; games never mix them)
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    c.execute(f"""CREATE TABLE ir AS SELECT r.uid, r.game_id, r.game_time, r.cat_id * 10 + coalesce(l.ru, 0) AS cat_id,
                  r.question_id, r.y::INT AS y, r.exposure, r.high_volume::INT AS high_volume, r.history_complete
                  FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id)""")
    c.execute(f"""CREATE TABLE side AS SELECT id AS game_id, user_id FROM
                  read_csv('{os.path.join(export, 'games.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    # gap bins: time since the learner's last encounter with the item (seen_* = sum over bins)
    # item difficulty (overall p-correct) and prior learner accuracy (KC level, games strictly before)
    c.execute(f"CREATE TABLE itp AS SELECT question_id, p_correct FROM {P('items')}")
    c.execute(f"""CREATE TABLE abil AS
        WITH g AS (SELECT uid, game_id, min(game_time) t, sum(y::INT) s, count(*) n FROM {P('kc_responses')} GROUP BY ALL)
        SELECT uid, game_id,
               sum(s) OVER w / nullif(sum(n) OVER w, 0) AS prior_acc,
               coalesce(count(*) OVER w, 0) AS prior_games
        FROM g WINDOW w AS (PARTITION BY uid ORDER BY t, game_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)""")

    # sample of responses (seeded, by hash so it is reproducible)
    c.execute(f"""CREATE TABLE s AS SELECT row_number() OVER () AS rid, r.*,
                  (r.uid = sd.user_id)::INT AS initiator, a.prior_acc, a.prior_games
                  FROM ir r JOIN side sd USING (game_id) LEFT JOIN abil a USING (uid, game_id)
                  WHERE hash(r.uid, r.game_id, r.question_id, {SEED}) % 1000 < {int(frac * 1000)}""")
    print("sampled responses:", c.execute("SELECT count(*) FROM s").fetchone()[0], flush=True)

    # learner history before each sampled response: per item, last exposure before t (same category)
    c.execute("""CREATE TABLE hist AS
        SELECT s.rid, e.question_id, arg_max(e.y, e.game_time) AS last_y, max(e.game_time) AS last_t
        FROM s JOIN ir e ON e.uid = s.uid AND e.cat_id = s.cat_id AND e.game_time < s.game_time
        GROUP BY s.rid, e.question_id""")
    print("history rows:", c.execute("SELECT count(*) FROM hist").fetchone()[0], flush=True)

    # observed properties of the drawn item
    c.execute("""CREATE TABLE obs AS
        SELECT s.rid, (h.question_id IS NOT NULL)::INT AS seen, coalesce((h.last_y = 0)::INT, 0) AS prev_wrong,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 0 AND date_diff('second', h.last_t, s.game_time) < 3600)::INT, 0) AS gap_lt1h,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 3600 AND date_diff('second', h.last_t, s.game_time) < 21600)::INT, 0) AS gap_1_6h,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 21600 AND date_diff('second', h.last_t, s.game_time) < 86400)::INT, 0) AS gap_6_24h,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 86400 AND date_diff('second', h.last_t, s.game_time) < 259200)::INT, 0) AS gap_1_3d,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 259200 AND date_diff('second', h.last_t, s.game_time) < 604800)::INT, 0) AS gap_3_7d,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 604800 AND date_diff('second', h.last_t, s.game_time) < 1209600)::INT, 0) AS gap_7_14d,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 1209600 AND date_diff('second', h.last_t, s.game_time) < 2592000)::INT, 0) AS gap_14_30d,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 2592000 AND date_diff('second', h.last_t, s.game_time) < 7776000)::INT, 0) AS gap_30_90d,
               coalesce((date_diff('second', h.last_t, s.game_time) >= 7776000 AND date_diff('second', h.last_t, s.game_time) < 10000000000)::INT, 0) AS gap_gt90d
        FROM s LEFT JOIN hist h ON h.rid = s.rid AND h.question_id = s.question_id""")

    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, cat_id, question_id FROM ir")
    frames = []
    for win in ("week", "month", "quarter"):
        c.execute(f"""CREATE OR REPLACE TABLE pool AS
            SELECT cat_id, date_trunc('{win}', game_time) w, question_id, count(*) n FROM draws GROUP BY ALL""")
        c.execute(f"""CREATE OR REPLACE TABLE pool_tot AS SELECT cat_id, w, sum(n) n_tot FROM pool GROUP BY ALL""")
        # leave-learner-out: remove all draws of the learner's own games in the window
        c.execute(f"""CREATE OR REPLACE TABLE own AS SELECT r.uid, d.cat_id, date_trunc('{win}', d.game_time) w,
                         d.question_id, count(*) n_own
                         FROM draws d JOIN (SELECT DISTINCT uid, game_id FROM ir) r USING (game_id) GROUP BY ALL""")
        c.execute(f"""CREATE OR REPLACE TABLE own_tot AS SELECT uid, cat_id, w, sum(n_own) n_own_tot FROM own GROUP BY ALL""")
        c.execute(f"""CREATE OR REPLACE TABLE expct AS
            WITH sw AS (SELECT s.rid, s.uid, s.question_id AS drawn, t.n_tot - ot.n_own_tot AS denom, s.cat_id,
                               date_trunc('{win}', s.game_time) w, s.game_time FROM s
                        JOIN pool_tot t ON t.cat_id = s.cat_id AND t.w = date_trunc('{win}', s.game_time)
                        JOIN own_tot ot ON ot.uid = s.uid AND ot.cat_id = s.cat_id AND ot.w = date_trunc('{win}', s.game_time))
            SELECT sw.rid,
                   sum(p.n - coalesce(o.n_own, 0)) / any_value(sw.denom) AS e_seen,
                   sum((p.n - coalesce(o.n_own, 0)) * (h.last_y = 0)::INT) / any_value(sw.denom) AS e_prev_wrong,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 0 AND date_diff('second', h.last_t, sw.game_time) < 3600)::INT) / any_value(sw.denom) AS e_gap_lt1h,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 3600 AND date_diff('second', h.last_t, sw.game_time) < 21600)::INT) / any_value(sw.denom) AS e_gap_1_6h,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 21600 AND date_diff('second', h.last_t, sw.game_time) < 86400)::INT) / any_value(sw.denom) AS e_gap_6_24h,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 86400 AND date_diff('second', h.last_t, sw.game_time) < 259200)::INT) / any_value(sw.denom) AS e_gap_1_3d,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 259200 AND date_diff('second', h.last_t, sw.game_time) < 604800)::INT) / any_value(sw.denom) AS e_gap_3_7d,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 604800 AND date_diff('second', h.last_t, sw.game_time) < 1209600)::INT) / any_value(sw.denom) AS e_gap_7_14d,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 1209600 AND date_diff('second', h.last_t, sw.game_time) < 2592000)::INT) / any_value(sw.denom) AS e_gap_14_30d,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 2592000 AND date_diff('second', h.last_t, sw.game_time) < 7776000)::INT) / any_value(sw.denom) AS e_gap_30_90d,
                   sum((p.n - coalesce(o.n_own, 0)) * (date_diff('second', h.last_t, sw.game_time) >= 7776000 AND date_diff('second', h.last_t, sw.game_time) < 10000000000)::INT) / any_value(sw.denom) AS e_gap_gt90d
            FROM sw JOIN hist h ON h.rid = sw.rid AND sw.denom > 0
            JOIN pool p ON p.cat_id = sw.cat_id AND p.w = sw.w AND p.question_id = h.question_id
            LEFT JOIN own o ON o.uid = sw.uid AND o.cat_id = sw.cat_id AND o.w = sw.w AND o.question_id = h.question_id
            GROUP BY sw.rid""")
        c.execute(f"""CREATE OR REPLACE TABLE pool_d AS
            SELECT p.cat_id, p.w, sum(p.n * i.p_correct) / sum(p.n) AS pool_p FROM pool p JOIN itp i USING (question_id) GROUP BY ALL""")
        f = c.sql(f"""
            SELECT s.rid, s.uid, s.initiator, s.high_volume, s.history_complete, s.prior_acc, s.prior_games,
                   s.game_time >= TIMESTAMP '2026-04-01' AS post_april, year(s.game_time) * 10 + quarter(s.game_time) AS yq,
                   o.seen, o.prev_wrong, o.gap_lt1h, o.gap_1_6h, o.gap_6_24h, o.gap_1_3d, o.gap_3_7d, o.gap_7_14d, o.gap_14_30d, o.gap_30_90d, o.gap_gt90d,
                   coalesce(e.e_seen, 0) e_seen, coalesce(e.e_prev_wrong, 0) e_prev_wrong,
                   coalesce(e.e_gap_lt1h, 0) e_gap_lt1h, coalesce(e.e_gap_1_6h, 0) e_gap_1_6h, coalesce(e.e_gap_6_24h, 0) e_gap_6_24h, coalesce(e.e_gap_1_3d, 0) e_gap_1_3d, coalesce(e.e_gap_3_7d, 0) e_gap_3_7d, coalesce(e.e_gap_7_14d, 0) e_gap_7_14d, coalesce(e.e_gap_14_30d, 0) e_gap_14_30d, coalesce(e.e_gap_30_90d, 0) e_gap_30_90d, coalesce(e.e_gap_gt90d, 0) e_gap_gt90d,
                   i.p_correct - d.pool_p AS d_minus_pool, '{win}' AS window
            FROM s JOIN obs o USING (rid) LEFT JOIN expct e USING (rid)
            JOIN itp i ON i.question_id = s.question_id
            JOIN pool_d d ON d.cat_id = s.cat_id AND d.w = date_trunc('{win}', s.game_time)
            JOIN pool_tot t ON t.cat_id = s.cat_id AND t.w = date_trunc('{win}', s.game_time)
            JOIN own_tot ot ON ot.uid = s.uid AND ot.cat_id = s.cat_id AND ot.w = date_trunc('{win}', s.game_time)
            WHERE t.n_tot - ot.n_own_tot > 0""").df()
        frames.append(f)
        print(win, "done", flush=True)
    df = pd.concat(frames, ignore_index=True)

    rng = np.random.default_rng(SEED)
    props = ["seen", "prev_wrong"] + [f"gap_{b}" for b in ("lt1h", "1_6h", "6_24h", "1_3d", "3_7d", "7_14d", "14_30d", "30_90d", "gt90d")]

    def summarise(d, label, by_window=True):
        rows = []
        for win, g in d.groupby("window", sort=False):
            uniq, inv = np.unique(g.uid.values, return_inverse=True)
            for p in props:
                diff = (g[p] - g["e_" + p]).values
                s_u = np.bincount(inv, diff); n_u = np.bincount(inv)
                boots = []
                for _ in range(N_BOOT):
                    k = rng.integers(0, len(uniq), len(uniq))
                    boots.append(s_u[k].sum() / n_u[k].sum())
                lo, hi = np.percentile(boots, [2.5, 97.5])
                obs, exp = g[p].mean(), g["e_" + p].mean()
                rows.append(dict(subset=label, window=win, property=p, n=len(g), learners=len(uniq),
                                 observed=obs, expected=exp, diff=obs - exp, ci_low=lo, ci_high=hi,
                                 rel_diff=(obs - exp) / exp if exp > 0 else np.nan))
        return rows

    rows = summarise(df, "all")
    m = df[df.window == "month"]
    for lab, mask in [("initiator", m.initiator == 1), ("opponent", m.initiator == 0),
                      ("pre 2026-04", ~m.post_april), ("post 2026-04", m.post_april),
                      ("high_volume", m.high_volume == 1), ("not high_volume", m.high_volume == 0),
                      ("cohort 2025+", m.history_complete == 1)]:
        rows += summarise(m[mask], lab)
    for q, g in m.groupby("yq"):
        rows += summarise(g, f"quarter {q // 10}Q{q % 10}")
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(out, "randomization_balance.csv"), index=False, float_format="%.5f")

    # difficulty of the drawn item vs prior learner accuracy (month window); slope should be ~0
    d = m.dropna(subset=["prior_acc", "d_minus_pool"])
    X = np.column_stack([np.ones(len(d)), d.prior_acc.values, np.log1p(d.prior_games.values)])
    beta = np.linalg.lstsq(X, d.d_minus_pool.values, rcond=None)[0]
    uniq, inv = np.unique(d.uid.values, return_inverse=True)
    order = np.argsort(inv); starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    bs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(uniq), len(uniq))
        rr = np.concatenate([order[starts[g]:ends[g]] for g in pick])
        bs.append(np.linalg.lstsq(X[rr], d.d_minus_pool.values[rr], rcond=None)[0])
    bs = np.array(bs)
    reg = pd.DataFrame({"term": ["intercept", "prior_acc", "log1p_prior_games"], "coef": beta,
                        "ci_low": np.percentile(bs, 2.5, axis=0), "ci_high": np.percentile(bs, 97.5, axis=0)})
    reg["outcome_sd"] = d.d_minus_pool.std()
    reg.to_csv(os.path.join(out, "difficulty_vs_ability.csv"), index=False, float_format="%.5f")

    pd.set_option("display.width", 220)
    print(res[res.subset == "all"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(res[(res.subset != "all") & (res.property.isin(["seen", "prev_wrong"]))]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(reg.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.1)
