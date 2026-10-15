"""
RQ1b: time-based decay vs interference from intervening practice.

    python rq1b_decay_interference.py <clean_dir> <export_dir> <out_dir> [--refine]

--refine: E1 only, P3/P4/T re-estimated with effect-modifier controls seen*log1p(experience) and
seen*log(learner intensity). Learner FE do not remove heterogeneity of the re-exposure EFFECT across
learners: intense players accumulate many games between encounters and may also gain less from a
re-exposure, so 'games since' could proxy learner type rather than interference.
    experience = all games of the learner before this game (prior_games_all)
    intensity  = learner's games per active day over the export (activity_days)

Same identification and estimator as rq1_causal.py (E1: learner x game FE + item FE, LPM,
global-recency controls, learner-clustered SE; E3 as check). Among re-exposures the time since the
last encounter and the amount of intervening practice vary independently enough (learners play at
different intensities), so both can enter jointly:
  P  parametric: seen, seen*log1p(hours), seen*log1p(games), seen*log1p(category games)
     games          = all games of the learner since the last encounter with the item
     category games = games since then in which the learner played the item's category (any round)
  T  cells: time bin x games bin (reference: unseen)
If at fixed time the effect falls with games -> interference; if at fixed games it falls with time -> decay.
Period: games before 2026-04-01. Only aggregates are written.
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
GB = [(1, 11, "g1-10"), (11, 51, "g11-50"), (51, 201, "g51-200"), (201, 10**9, "g>200")]


def main(clean, export, out, refine=False):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    # category-game index: games in which the learner played the category in any round
    c.execute(f"""CREATE TABLE catg AS
        SELECT uid, cat_id, game_id, row_number() OVER (PARTITION BY uid, cat_id ORDER BY t, game_id) AS cat_game_idx
        FROM (SELECT uid, cat_id, game_id, min(game_time) t FROM {P('kc_responses')} GROUP BY ALL)""")
    c.execute(f"""CREATE TABLE ir AS
        SELECT r.uid, r.game_id, r.pos, r.game_time, r.cat_id * 10 + coalesce(l.ru, 0) AS pk, r.question_id, r.y::INT AS y,
               r.exposure, r.gap_seconds, cg.cat_game_idx, r.prior_games_all,
               r.prior_games_all - lag(r.prior_games_all) OVER w AS games_since,
               cg.cat_game_idx - lag(cg.cat_game_idx) OVER w AS cat_games_since
        FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id)
        JOIN catg cg ON cg.uid = r.uid AND cg.cat_id = r.cat_id AND cg.game_id = r.game_id
        WHERE r.game_time < TIMESTAMP '{END}'
        WINDOW w AS (PARTITION BY r.uid, r.question_id ORDER BY r.game_time, r.game_id)""")
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, pk, question_id FROM ir")
    c.execute("""CREATE TABLE grec AS SELECT i.uid, i.game_id, i.pos, date_diff('second', d.game_time, i.game_time) AS g_rec
                 FROM ir i ASOF LEFT JOIN draws d ON d.pk = i.pk AND d.question_id = i.question_id AND d.game_time < i.game_time""")
    c.execute("CREATE TABLE idiff AS SELECT question_id, sum(y) s, count(*) n FROM ir WHERE exposure = 1 GROUP BY 1")
    c.execute("CREATE TABLE udiff AS SELECT uid, question_id, sum(y) s, count(*) n FROM ir WHERE exposure = 1 GROUP BY ALL")
    df = c.sql(f"""
        SELECT i.uid, i.game_id, i.question_id, i.y, (i.exposure > 1)::INT AS seen, i.prior_games_all AS experience, a.intensity,
               coalesce(i.gap_seconds / 3600.0, 0) AS hours, coalesce(i.games_since, 0) AS games,
               coalesce(i.cat_games_since, 0) AS cat_games,
               {bins_case('i.gap_seconds', TB)} AS tb, {bins_case('i.games_since', GB)} AS gb,
               {bins_case('g.g_rec', GREC, "'g_never'")} AS grec_bin,
               (d.s - coalesce(u.s, 0)) / nullif(d.n - coalesce(u.n, 0), 0) AS item_p_loo
        FROM ir i JOIN grec g USING (uid, game_id, pos)
        JOIN idiff d USING (question_id) LEFT JOIN udiff u USING (uid, question_id)
        JOIN (SELECT uid, sum(games) / count(*) AS intensity FROM {P('activity_days')} GROUP BY uid) a USING (uid)""").df()
    seen = df.seen.values.astype(float)
    print("responses:", len(df), "re-exposures:", int(seen.sum()), flush=True)
    s_ = df[df.seen == 1]
    corr = np.corrcoef(np.log1p(s_[["hours", "games", "cat_games"]].values.T))
    print("corr of log1p(hours, games, cat_games) among re-exposures:\n", np.round(corr, 3), flush=True)

    y = df.y.values.astype(float)
    g_game = pd.factorize(df.uid.astype(str) + "_" + df.game_id.astype(str))[0]
    g_item = pd.factorize(df.question_id)[0]
    cluster = df.uid.values
    gcols = [(df.grec_bin.values == g).astype(float) for g in ["g<1h", "g1-24h", "g1-7d", "g_never"]]
    gnames = ["ctrl " + g for g in ["g<1h", "g1-24h", "g1-7d", "g_never"]]
    lh, lg, lc = (seen * np.log1p(df[v].values) for v in ("hours", "games", "cat_games"))

    specs = {
        "P1 time only": ([seen, lh], ["seen", "seen x log1p(hours)"]),
        "P2 games only": ([seen, lg], ["seen", "seen x log1p(games)"]),
        "P3 time + games": ([seen, lh, lg], ["seen", "seen x log1p(hours)", "seen x log1p(games)"]),
        "P4 time + games + category games": ([seen, lh, lg, lc],
                                             ["seen", "seen x log1p(hours)", "seen x log1p(games)", "seen x log1p(cat games)"]),
    }
    cells, cnames = [], []
    for _, _, t in TB:
        for _, _, g in GB:
            cells.append(((df.tb.values == t) & (df.gb.values == g)).astype(float)); cnames.append(f"{t} | {g}")
    specs["T time x games cells"] = (cells, cnames)

    if refine:
        mod = [seen * np.log1p(df.experience.values), seen * np.log(df.intensity.values)]
        mnames = ["seen x log1p(experience)", "seen x log(intensity)"]
        specs = {k + " + modifiers": (v[0] + mod, v[1] + mnames)
                 for k, v in specs.items() if k.startswith(("P3", "P4", "T"))}
    item_p = np.nan_to_num(df.item_p_loo.values, nan=np.nanmean(df.item_p_loo.values))
    rows, info = [], []
    for sname, (cols, names) in specs.items():
        ests = [("E1 learner-game FE + item FE", [g_game, g_item], [])]
        if not refine:
            ests.append(("E3 learner-game FE + item difficulty", [g_game], [item_p]))
        for ename, groups, extra in ests:
            X = np.column_stack(cols + gcols + extra)
            res, meta = fe_ols(y, X, names + gnames + (["item_p_loo"] if extra else []), groups, cluster)
            res.insert(0, "estimator", ename); res.insert(0, "spec", sname)
            rows.append(res); info.append(dict(spec=sname, estimator=ename, **meta))
            print(sname, "|", ename, meta, flush=True)
    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(os.path.join(out, "rq1b_decay_interference" + ("_refined" if refine else "") + ".csv"),
                  index=False, float_format="%.5f")
    cnt = s_.groupby(["tb", "gb"]).size().rename("n").reset_index()
    cnt.to_csv(os.path.join(out, "rq1b_cell_counts.csv"), index=False)
    pd.DataFrame(corr, index=["hours", "games", "cat_games"], columns=["hours", "games", "cat_games"]).to_csv(
        os.path.join(out, "rq1b_corr.csv"), float_format="%.3f")
    pd.set_option("display.width", 220)
    show = out_df[~out_df.term.str.startswith("ctrl") & (out_df.term != "item_p_loo")]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(cnt.pivot(index="gb", columns="tb", values="n").to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], refine="--refine" in sys.argv[4:])
