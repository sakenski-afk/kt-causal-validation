"""
RQ1c: sensitivity of the post-error / post-correct contrast to selection.

    python rq1c_error_sensitivity.py <clean_dir> <export_dir> <out_dir>

The outcome of the previous encounter is not randomized: an item a learner got wrong tends to be
hard for that learner (learner-item specific difficulty, not absorbed by item FE). Two checks:
 1) Expectedness strata. For the previous encounter we compute its expected correctness from
    information available before it: the learner's smoothed accuracy in the category up to that game
    (KC level) and the item's leave-learner-out first-exposure p-correct (Rasch-type combination on
    the logit scale). Previous outcomes are split by terciles of this expectation. An UNEXPECTED
    error (high expected correctness) is closer to a random slip, so selection is weaker there; if
    the decay pattern after unexpected errors matches the pooled one, the conclusion is robust.
 2) Asymptote decomposition (in the report): with the feedback benefit decayed at long lags, the
    long-lag coefficient approximates the persistent selection component.
Estimator: E1 (learner x game FE + item FE), LPM, global-recency controls, learner-clustered SE.
Lag bins are coarse (<1d, 1-7d, 7-30d, >30d) to keep cells populated. Period: before 2026-04-01.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_causal import bins_case, fe_ols, GREC  # noqa: E402

END = "2026-04-01"
TB = [(0, 86400, "<1d"), (86400, 7 * 86400, "1-7d"), (7 * 86400, 30 * 86400, "7-30d"), (30 * 86400, 10**10, ">30d")]


def main(clean, export, out):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    # learner accuracy in the category strictly before each game (KC level), smoothed towards the global mean
    c.execute(f"""CREATE TABLE catacc AS
        WITH g AS (SELECT uid, cat_id, game_id, min(game_time) t, sum(y::INT) s, count(*) n FROM {P('kc_responses')}
                   WHERE game_time < TIMESTAMP '{END}' GROUP BY ALL),
             m AS (SELECT avg(y::INT) mu FROM {P('kc_responses')} WHERE game_time < TIMESTAMP '{END}')
        SELECT uid, cat_id, game_id,
               (coalesce(sum(s) OVER w, 0) + 5 * (SELECT mu FROM m)) / (coalesce(sum(n) OVER w, 0) + 5) AS acc_cat
        FROM g WINDOW w AS (PARTITION BY uid, cat_id ORDER BY t, game_id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)""")
    c.execute(f"""CREATE TABLE ir AS
        SELECT r.uid, r.game_id, r.pos, r.game_time, r.cat_id, r.cat_id * 10 + coalesce(l.ru, 0) AS pk, r.question_id,
               r.y::INT AS y, r.exposure, r.prev_y, r.gap_seconds, a.acc_cat
        FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id)
        JOIN catacc a ON a.uid = r.uid AND a.cat_id = r.cat_id AND a.game_id = r.game_id
        WHERE r.game_time < TIMESTAMP '{END}'""")
    c.execute("CREATE TABLE idiff AS SELECT question_id, sum(y) s, count(*) n FROM ir WHERE exposure = 1 GROUP BY 1")
    c.execute("CREATE TABLE udiff AS SELECT uid, question_id, sum(y) s, count(*) n FROM ir WHERE exposure = 1 GROUP BY ALL")
    c.execute("""CREATE TABLE ir2 AS
        SELECT i.*, (d.s - coalesce(u.s, 0) + 1) / (d.n - coalesce(u.n, 0) + 2) AS item_p
        FROM ir i JOIN idiff d USING (question_id) LEFT JOIN udiff u USING (uid, question_id)""")
    # expected correctness at each encounter (Rasch-type), then the value at the PREVIOUS encounter
    c.execute("""CREATE TABLE ir3 AS
        WITH e AS (SELECT *, 1 / (1 + exp(-(ln(acc_cat / (1 - acc_cat)) + ln(item_p / (1 - item_p))
                                               - ln((SELECT avg(y) FROM ir) / (1 - (SELECT avg(y) FROM ir)))))) AS p_exp
                   FROM ir2)
        SELECT *, lag(p_exp) OVER (PARTITION BY uid, question_id ORDER BY game_time, game_id) AS p_prev FROM e""")
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, pk, question_id FROM ir3")
    c.execute("""CREATE TABLE grec AS SELECT i.uid, i.game_id, i.pos, date_diff('second', d.game_time, i.game_time) AS g_rec
                 FROM ir3 i ASOF LEFT JOIN draws d ON d.pk = i.pk AND d.question_id = i.question_id AND d.game_time < i.game_time""")
    df = c.sql(f"""
        SELECT i.uid, i.game_id, i.question_id, i.y, (i.exposure > 1)::INT AS seen, coalesce(i.prev_y, -1) AS prev_y,
               i.p_prev, {bins_case('i.gap_seconds', TB)} AS tb, {bins_case('g.g_rec', GREC, "'g_never'")} AS grec_bin
        FROM ir3 i JOIN grec g USING (uid, game_id, pos)""").df()
    print("responses:", len(df), "re-exposures:", int(df.seen.sum()), flush=True)

    # terciles of expected correctness of the previous encounter, separately after errors and after correct answers
    df["tert"] = None
    for pv in (0, 1):
        m = df.prev_y == pv
        q = df.loc[m, "p_prev"].quantile([1 / 3, 2 / 3]).values
        df.loc[m, "tert"] = np.where(df.loc[m, "p_prev"] < q[0], "T1 low", np.where(df.loc[m, "p_prev"] < q[1], "T2 mid", "T3 high"))
        print("prev_y", pv, "tercile cut points of expected correctness:", np.round(q, 3), flush=True)
    strata = df[df.seen == 1].groupby(["prev_y", "tert"]).agg(n=("y", "size"), mean_p_prev=("p_prev", "mean"),
                                                             mean_y=("y", "mean")).reset_index()

    y = df.y.values.astype(float)
    g_game = pd.factorize(df.uid.astype(str) + "_" + df.game_id.astype(str))[0]
    g_item = pd.factorize(df.question_id)[0]
    gcols = [(df.grec_bin.values == g).astype(float) for g in ["g<1h", "g1-24h", "g1-7d", "g_never"]]
    gnames = ["ctrl " + g for g in ["g<1h", "g1-24h", "g1-7d", "g_never"]]
    cols, names = [], []
    for pv, lab in ((1, "after correct"), (0, "after error")):
        for t in ("T1 low", "T2 mid", "T3 high"):
            for _, _, b in TB:
                cols.append(((df.prev_y.values == pv) & (df.tert.values == t) & (df.tb.values == b)).astype(float))
                names.append(f"{lab} | exp {t} | {b}")
    res, meta = fe_ols(y, np.column_stack(cols + gcols), names + gnames, [g_game, g_item], df.uid.values)
    print(meta, flush=True)
    res.to_csv(os.path.join(out, "rq1c_error_sensitivity.csv"), index=False, float_format="%.5f")
    strata.to_csv(os.path.join(out, "rq1c_strata.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print(res[~res.term.str.startswith("ctrl")].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(strata.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
