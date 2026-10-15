"""
Reviewer check: balance of the round-5 item draw by learner strata, and positivity.

    python review_balance.py <clean_dir> <export_dir> <out_dir> [sample_frac]

Same test as randomization_check.py (observed vs leave-learner-out expected share of drawn items with a
learner-specific property; calendar-month window; category x language pools), restricted to the analysis period
(games before 2026-04-01) and reported by
  - learner activity tercile (games per active day),
  - experience (number of earlier games),
  - item language and the ten largest categories (others pooled),
  - quarter.
Positivity: distribution of the expected probability that a draw is a previously seen item, e_seen, by experience;
the within-game contrast needs 0 < e_seen < 1 (both statuses possible for the same learner state).
Only aggregates are written.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

SEED, N_BOOT, END = 42, 500, "2026-04-01"


def main(clean, export, out, frac=0.1):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute(f"SET temp_directory='{os.path.join(out, '_duckdb_tmp')}'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    c.execute(f"""CREATE TABLE ir AS SELECT r.uid, r.game_id, r.game_time, r.cat_id AS cat, r.cat_id * 10 + coalesce(l.ru, 0) AS pk,
                  coalesce(l.ru, 0) AS ru, r.question_id, r.y::INT AS y, r.exposure, r.prior_games_all
                  FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id) WHERE r.game_time < TIMESTAMP '{END}'""")
    c.execute(f"""CREATE TABLE act AS SELECT uid, sum(games) / count(*) AS gpd FROM {P('activity_days')}
                  WHERE d < DATE '{END}' GROUP BY 1""")
    c.execute(f"""CREATE TABLE s AS SELECT row_number() OVER () AS rid, r.*, a.gpd FROM ir r LEFT JOIN act a USING (uid)
                  WHERE hash(r.uid, r.game_id, r.question_id, {SEED}) % 1000 < {int(frac * 1000)}""")
    print("sampled responses:", c.execute("SELECT count(*) FROM s").fetchone()[0], flush=True)
    c.execute("""CREATE TABLE hist AS SELECT s.rid, e.question_id, arg_max(e.y, e.game_time) AS last_y
                 FROM s JOIN ir e ON e.uid = s.uid AND e.pk = s.pk AND e.game_time < s.game_time GROUP BY s.rid, e.question_id""")
    c.execute("""CREATE TABLE obs AS SELECT s.rid, (h.question_id IS NOT NULL)::INT AS seen, coalesce((h.last_y = 0)::INT, 0) AS prev_wrong
                 FROM s LEFT JOIN hist h ON h.rid = s.rid AND h.question_id = s.question_id""")
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, pk, question_id FROM ir")
    c.execute("CREATE TABLE pool AS SELECT pk, date_trunc('month', game_time) w, question_id, count(*) n FROM draws GROUP BY ALL")
    c.execute("CREATE TABLE pool_tot AS SELECT pk, w, sum(n) n_tot FROM pool GROUP BY ALL")
    c.execute("""CREATE TABLE own AS SELECT r.uid, d.pk, date_trunc('month', d.game_time) w, d.question_id, count(*) n_own
                 FROM draws d JOIN (SELECT DISTINCT uid, game_id FROM ir) r USING (game_id) GROUP BY ALL""")
    c.execute("CREATE TABLE own_tot AS SELECT uid, pk, w, sum(n_own) n_own_tot FROM own GROUP BY ALL")
    c.execute("""CREATE TABLE expct AS
        WITH sw AS (SELECT s.rid, s.uid, s.pk, date_trunc('month', s.game_time) w, t.n_tot - ot.n_own_tot AS denom FROM s
                    JOIN pool_tot t ON t.pk = s.pk AND t.w = date_trunc('month', s.game_time)
                    JOIN own_tot ot ON ot.uid = s.uid AND ot.pk = s.pk AND ot.w = date_trunc('month', s.game_time))
        SELECT sw.rid, any_value(sw.denom) AS denom,
               sum(p.n - coalesce(o.n_own, 0)) / any_value(sw.denom) AS e_seen,
               sum((p.n - coalesce(o.n_own, 0)) * (h.last_y = 0)::INT) / any_value(sw.denom) AS e_prev_wrong
        FROM sw LEFT JOIN hist h ON h.rid = sw.rid
        LEFT JOIN pool p ON p.pk = sw.pk AND p.w = sw.w AND p.question_id = h.question_id
        LEFT JOIN own o ON o.uid = sw.uid AND o.pk = sw.pk AND o.w = sw.w AND o.question_id = h.question_id
        WHERE sw.denom > 0 GROUP BY sw.rid""")
    df = c.sql("""SELECT s.rid, s.uid, s.cat, s.ru, s.gpd, s.prior_games_all, year(s.game_time) * 10 + quarter(s.game_time) AS yq,
                         o.seen, o.prev_wrong, coalesce(e.e_seen, 0) AS e_seen, coalesce(e.e_prev_wrong, 0) AS e_prev_wrong
                  FROM s JOIN obs o USING (rid) JOIN expct e USING (rid)""").df()
    print("responses with expectation:", len(df), flush=True)

    terc = pd.qcut(df.groupby("uid").gpd.first(), 3, labels=["low", "mid", "high"])
    df["activity"] = "activity " + df.uid.map(terc).astype(str)
    df["experience"] = "experience " + pd.cut(df.prior_games_all, [-1, 10, 50, 200, 1000, 10**9],
                                              labels=["0-10", "11-50", "51-200", "201-1000", ">1000"]).astype(str)
    top = df.cat.value_counts().index[:10]
    df["category"] = np.where(df.cat.isin(top), "category " + df.cat.astype(str), "category other")
    df["language"] = np.where(df.ru == 1, "language ru", "language kk")
    df["quarter"] = "quarter " + (df.yq // 10).astype(str) + "Q" + (df.yq % 10).astype(str)

    rng = np.random.default_rng(SEED)
    rows = []
    for dim in ("all", "activity", "experience", "language", "category", "quarter"):
        groups = [("all", df)] if dim == "all" else list(df.groupby(dim))
        for lab, g in groups:
            uniq, inv = np.unique(g.uid.values, return_inverse=True)
            for prop in ("seen", "prev_wrong"):
                diff = (g[prop] - g["e_" + prop]).values
                s_u, n_u = np.bincount(inv, diff), np.bincount(inv)
                boots = [s_u[k].sum() / n_u[k].sum() for k in (rng.integers(0, len(uniq), len(uniq)) for _ in range(N_BOOT))]
                lo, hi = np.percentile(boots, [2.5, 97.5])
                o, e = g[prop].mean(), g["e_" + prop].mean()
                rows.append(dict(dimension=dim, stratum=lab, property=prop, n=len(g), learners=len(uniq), observed=o, expected=e,
                                 diff=o - e, ci_low=lo, ci_high=hi, rel_diff=(o - e) / e if e > 0 else np.nan))
    bal = pd.DataFrame(rows)
    bal.to_csv(os.path.join(out, "balance_by_strata.csv"), index=False, float_format="%.5f")

    pos = df.groupby("experience").e_seen.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    pos["share_strictly_between_0_and_1"] = df.groupby("experience").e_seen.apply(lambda v: ((v > 0.001) & (v < 0.999)).mean())
    pos["share_above_0_9"] = df.groupby("experience").e_seen.apply(lambda v: (v > 0.9).mean())
    pos["observed_seen_share"] = df.groupby("experience").seen.mean()
    pos.to_csv(os.path.join(out, "positivity_by_experience.csv"), float_format="%.4f")

    pd.set_option("display.width", 220)
    print(bal.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pos.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.1)
