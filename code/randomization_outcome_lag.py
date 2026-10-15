"""
Design check for the post-error contrast: does the timing of a re-exposure depend on the previous outcome?

    python randomization_outcome_lag.py <clean_dir> <export_dir> <out_dir> [sample_frac]

The lag curves after a correct and after a wrong previous answer (specification C) are interpreted by their
shape. This requires that, given the pool and global item rotation, the probability that an item the learner
has seen is drawn again at lag L does not depend on whether the learner answered it correctly. Same design as
randomization_check.py (category x item-language pools, month window, leave-learner-out draw shares,
period before 2026-04-01): for every sampled response and every item the learner has seen in that pool,
observed draw (0/1) and expected draw share are cross-tabulated by (previous outcome, lag bin).
ratio = observed / expected; the check is that the ratios after wrong and after correct answers coincide
within each lag bin (difference and 95% CI by learner-clustered bootstrap). Only aggregates are written.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 300
END = "2026-04-01"
LAG = [(0, 3600, "<1h"), (3600, 6 * 3600, "1-6h"), (6 * 3600, 86400, "6-24h"), (86400, 3 * 86400, "1-3d"),
       (3 * 86400, 7 * 86400, "3-7d"), (7 * 86400, 14 * 86400, "7-14d"), (14 * 86400, 30 * 86400, "14-30d"),
       (30 * 86400, 90 * 86400, "30-90d"), (90 * 86400, 10**10, ">90d")]


def main(clean, export, out, frac=0.1):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute(f"SET temp_directory='{os.path.join(out, '_duckdb_tmp')}'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    c.execute(f"""CREATE TABLE ir AS SELECT r.uid, r.game_id, r.game_time, r.cat_id * 10 + coalesce(l.ru, 0) AS pk,
                  r.question_id, r.y::INT AS y
                  FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id) WHERE r.game_time < TIMESTAMP '{END}'""")
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, pk, question_id FROM ir")
    c.execute("CREATE TABLE pool AS SELECT pk, date_trunc('month', game_time) w, question_id, count(*) n FROM draws GROUP BY ALL")
    c.execute("CREATE TABLE pool_tot AS SELECT pk, w, sum(n) n_tot FROM pool GROUP BY ALL")
    c.execute("""CREATE TABLE own AS SELECT r.uid, d.pk, date_trunc('month', d.game_time) w, d.question_id, count(*) n_own
                 FROM draws d JOIN (SELECT DISTINCT uid, game_id FROM ir) r USING (game_id) GROUP BY ALL""")
    c.execute("CREATE TABLE own_tot AS SELECT uid, pk, w, sum(n_own) n_own_tot FROM own GROUP BY ALL")
    c.execute(f"""CREATE TABLE s AS SELECT row_number() OVER () rid, uid, game_id, game_time, pk, question_id AS drawn,
                  date_trunc('month', game_time) w FROM ir WHERE hash(uid, game_id, question_id, {SEED}) % 1000 < {int(frac * 1000)}""")
    c.execute("""CREATE TABLE hist AS SELECT s.rid, e.question_id, arg_max(e.y, e.game_time) last_y, max(e.game_time) last_t
                 FROM s JOIN ir e ON e.uid = s.uid AND e.pk = s.pk AND e.game_time < s.game_time GROUP BY ALL""")
    lag = "CASE " + " ".join(f"WHEN date_diff('second', h.last_t, s.game_time) >= {a} AND date_diff('second', h.last_t, s.game_time) < {b} THEN '{lab}'" for a, b, lab in LAG) + " END"
    cells = c.sql(f"""
        WITH x AS (
          SELECT s.uid, h.last_y, {lag} lb, (h.question_id = s.drawn)::INT AS obs,
                 (p.n - coalesce(o.n_own, 0)) / nullif(t.n_tot - ot.n_own_tot, 0) AS e
          FROM s JOIN hist h ON h.rid = s.rid
          JOIN pool p ON p.pk = s.pk AND p.w = s.w AND p.question_id = h.question_id
          JOIN pool_tot t ON t.pk = s.pk AND t.w = s.w
          JOIN own_tot ot ON ot.uid = s.uid AND ot.pk = s.pk AND ot.w = s.w
          LEFT JOIN own o ON o.uid = s.uid AND o.pk = s.pk AND o.w = s.w AND o.question_id = h.question_id)
        SELECT uid, last_y, lb, sum(obs) o, sum(e) e FROM x WHERE e IS NOT NULL AND lb IS NOT NULL GROUP BY ALL""").df()
    n_s = c.execute("SELECT count(*), count(DISTINCT uid) FROM s").fetchone()
    print("sampled responses / learners:", n_s, flush=True)

    rng = np.random.default_rng(SEED)
    uids = pd.Index(cells.uid.unique()); U = len(uids)
    labs = [lab for _, _, lab in LAG]
    O, E = np.zeros((2, U, len(labs))), np.zeros((2, U, len(labs)))
    ui = uids.get_indexer(cells.uid)
    for k, yv in enumerate((0, 1)):
        for j, l in enumerate(labs):
            m = (cells.last_y.values == yv) & (cells.lb.values == l)
            O[k, :, j] = np.bincount(ui[m], cells.o.values[m], minlength=U)
            E[k, :, j] = np.bincount(ui[m], cells.e.values[m], minlength=U)
    one = np.ones(U)
    r_err, r_cor = (one @ O[0]) / (one @ E[0]), (one @ O[1]) / (one @ E[1])
    B = np.zeros((N_BOOT, 3, len(labs)))
    for b in range(N_BOOT):
        w = rng.multinomial(U, np.full(U, 1 / U)).astype(float)
        re, rc = (w @ O[0]) / (w @ E[0]), (w @ O[1]) / (w @ E[1])
        B[b] = re, rc, re - rc
    lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
    res = pd.DataFrame({"lag_bin": labs,
                        "observed_after_error": (one @ O[0]).astype(int), "ratio_after_error": r_err, "lo_err": lo[0], "hi_err": hi[0],
                        "observed_after_correct": (one @ O[1]).astype(int), "ratio_after_correct": r_cor, "lo_cor": lo[1], "hi_cor": hi[1],
                        "difference": r_err - r_cor, "diff_lo": lo[2], "diff_hi": hi[2]})
    res.to_csv(os.path.join(out, "outcome_lag_balance.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.1)
