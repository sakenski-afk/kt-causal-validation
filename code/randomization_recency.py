"""
Which rule suppresses recent repeats: elapsed time or number of games since the last encounter?

    python randomization_recency.py <clean_dir> <export_dir> <out_dir> [sample_frac]

Same design as randomization_check.py (month window,
with leave-learner-out draw shares (see that script), but observed and expected re-exposure mass is cross-tabulated by
    time since the learner's last encounter with the item  x  learner games played since then
(prior_games_all counts all of the learner's games in the export, complete or not).
ratio = observed / expected; 1 means no selection. If the ratio varies with games at fixed time,
the rule counts games; if it varies with time at fixed games, the rule uses time.
Only aggregates are written.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 300
T_BINS = [(0, 3600, "<1h"), (3600, 86400, "1-24h"), (86400, 7 * 86400, "1-7d"),
          (7 * 86400, 30 * 86400, "7-30d"), (30 * 86400, 10**10, ">30d")]
G_BINS = [(1, 2, "1"), (2, 3, "2"), (3, 6, "3-5"), (6, 11, "6-10"), (11, 21, "11-20"),
          (21, 51, "21-50"), (51, 101, "51-100"), (101, 10**9, ">100")]


def case(expr, bins):
    return "CASE " + " ".join(f"WHEN {expr} >= {a} AND {expr} < {b} THEN '{lab}'" for a, b, lab in bins) + " END"


def main(clean, export, out, frac=0.1):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute(f"SET temp_directory='{os.path.join(out, '_duckdb_tmp')}'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    # pool key = category x item language (as in randomization_check.py)
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    c.execute(f"""CREATE TABLE ir AS SELECT r.uid, r.game_id, r.game_time, r.cat_id * 10 + coalesce(l.ru, 0) AS cat_id,
                  r.question_id, r.prior_games_all
                  FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id)
                  WHERE r.game_time < TIMESTAMP '2026-07-01'""")  # main analysis period
    c.execute(f"""CREATE TABLE s AS SELECT row_number() OVER () rid, * FROM ir
                  WHERE hash(uid, game_id, question_id, {SEED}) % 1000 < {int(frac * 1000)}""")
    c.execute("""CREATE TABLE hist AS
        SELECT s.rid, e.question_id, max(e.game_time) last_t, arg_max(e.prior_games_all, e.game_time) last_g
        FROM s JOIN ir e ON e.uid = s.uid AND e.cat_id = s.cat_id AND e.game_time < s.game_time
        GROUP BY ALL""")
    tcase = case("date_diff('second', h.last_t, s.game_time)", T_BINS)
    gcase = case("(s.prior_games_all - h.last_g)", G_BINS)
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, cat_id, question_id FROM ir")
    c.execute("""CREATE TABLE pool AS SELECT cat_id, date_trunc('month', game_time) w, question_id, count(*) n
                 FROM draws GROUP BY ALL""")
    c.execute("CREATE TABLE pool_tot AS SELECT cat_id, w, sum(n) n_tot FROM pool GROUP BY ALL")
    c.execute("""CREATE TABLE own AS SELECT r.uid, d.cat_id, date_trunc('month', d.game_time) w, d.question_id, count(*) n_own
                 FROM draws d JOIN (SELECT DISTINCT uid, game_id FROM ir) r USING (game_id) GROUP BY ALL""")
    c.execute("CREATE TABLE own_tot AS SELECT uid, cat_id, w, sum(n_own) n_own_tot FROM own GROUP BY ALL")
    # per sampled response and cell: observed indicator and expected share
    cells = c.sql(f"""
        WITH x AS (
          SELECT s.rid, s.uid, {tcase} tb, {gcase} gb,
                 (h.question_id = s.question_id)::INT AS obs,
                 (p.n - coalesce(o.n_own, 0)) / nullif(t.n_tot - ot.n_own_tot, 0) AS e
          FROM s JOIN hist h ON h.rid = s.rid
          JOIN pool p ON p.cat_id = s.cat_id AND p.w = date_trunc('month', s.game_time) AND p.question_id = h.question_id
          JOIN pool_tot t ON t.cat_id = s.cat_id AND t.w = date_trunc('month', s.game_time)
          JOIN own_tot ot ON ot.uid = s.uid AND ot.cat_id = s.cat_id AND ot.w = date_trunc('month', s.game_time)
          LEFT JOIN own o ON o.uid = s.uid AND o.cat_id = s.cat_id AND o.w = date_trunc('month', s.game_time)
                         AND o.question_id = h.question_id)
        SELECT uid, tb, gb, sum(obs) o, sum(e) e FROM x WHERE tb IS NOT NULL AND gb IS NOT NULL AND e IS NOT NULL GROUP BY ALL""").df()
    n_s = c.execute("SELECT count(*), count(DISTINCT uid) FROM s").fetchone()
    print("sampled responses / learners:", n_s, flush=True)

    rng = np.random.default_rng(SEED)
    rows = []
    for (tb, gb), g in cells.groupby(["tb", "gb"]):
        o, e = g.o.values, g.e.values
        boots = []
        for _ in range(N_BOOT):
            k = rng.integers(0, len(g), len(g))
            boots.append(o[k].sum() / e[k].sum())
        rows.append(dict(time_since=tb, games_since=gb, observed=o.sum(), expected=round(e.sum(), 1),
                         ratio=o.sum() / e.sum(), ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5)))
    res = pd.DataFrame(rows)
    tord = {lab: i for i, (_, _, lab) in enumerate(T_BINS)}
    gord = {lab: i for i, (_, _, lab) in enumerate(G_BINS)}
    res = res.sort_values(["time_since", "games_since"], key=lambda s: s.map(tord if s.name == "time_since" else gord))
    res.to_csv(os.path.join(out, "recency_rule_time_x_games.csv"), index=False, float_format="%.4f")
    piv = res.pivot(index="games_since", columns="time_since", values="ratio").reindex(
        [lab for _, _, lab in G_BINS])[[lab for _, _, lab in T_BINS]]
    cnt = res.pivot(index="games_since", columns="time_since", values="expected").reindex(
        [lab for _, _, lab in G_BINS])[[lab for _, _, lab in T_BINS]]
    print("ratio observed/expected (rows: games since last encounter, cols: time since)")
    print(piv.round(2).to_string())
    print("\nexpected count per cell")
    print(cnt.round(0).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.1)
