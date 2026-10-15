"""
RQ1: causal effects of re-exposure and of the interval since the last encounter on correctness.

    python rq1_causal.py <clean_dir> <export_dir> <out_dir> [--test-period]

--test-period: estimation sample restricted to the RQ2 test window 2026-01-01 .. 2026-03-31 (histories,
draw shares and global recency still use all earlier data); E1 only, specifications A-C; files *_test.csv.
This is the causal reference for comparing model-implied effects on the same responses.
--window START END (with --test-period): another test window, e.g. a rolling-origin fold; files *_test_<yyyy-mm>.csv.
--fine-lags: whole period, E1 only, specifications B-C with 14 finer lag bins (10 min .. >90 d) for the RQ3 review
policy; files *_fine.csv plus the geometric-mean lag of each bin (rq1_fine_lag_hours.csv).

Period: games before 2026-04-01 (data-owner decision). Item level = round 5 (3 items per game).

Identification (see work/design_checks/RANDOMIZATION_CHECK.md): which items a learner receives is
random with respect to the learner (no dependence on ability, past errors or personal repeat history);
the draw probability depends only on the item (pool share, global recency of its last deal).
Within one game the three items share the learner's state (ability, fatigue, tenure), so a
learner-game fixed effect removes all time-varying learner confounding; whether a drawn item was
seen before is then set by the random draw. Item fixed effects remove item difficulty; global recency
bins enter as controls because they shift the draw probability.

Estimators (linear probability models, high-dimensional FE absorbed exactly: within-group demeaning
for one grouping, LSMR on the sparse dummy design for two,
cluster-robust SE by learner):
  E1 learner x game FE + item FE                (primary)
  E2 learner x day FE  + item FE                (weaker time control, more within-group variation)
  E3 learner x game FE + item difficulty (leave-learner-out first-exposure p-correct), no item FE
Specifications:
  A  seen-before indicator
  B  time-since-last-encounter bins (reference: unseen)
  C  B split by the outcome of the last encounter (post-error / post-correct) - conditional contrast:
     the previous outcome is NOT randomized, so C is reported as conditional, not causal
  D  games-since-last-encounter bins
Note: item ids are stored only for round 5. Unrecorded encounters in rounds 1-4 hit seen and unseen
items alike (random draws), so effects are those of a recorded round-5 encounter (conservative).
Only aggregates are written.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import lsmr

END = "2026-04-01"
LAG = [(0, 3600, "<1h"), (3600, 6 * 3600, "1-6h"), (6 * 3600, 86400, "6-24h"), (86400, 3 * 86400, "1-3d"),
       (3 * 86400, 7 * 86400, "3-7d"), (7 * 86400, 14 * 86400, "7-14d"), (14 * 86400, 30 * 86400, "14-30d"),
       (30 * 86400, 90 * 86400, "30-90d"), (90 * 86400, 10**10, ">90d")]
GAMES = [(1, 3, "1-2"), (3, 11, "3-10"), (11, 51, "11-50"), (51, 201, "51-200"), (201, 10**9, ">200")]
FINE_LAG = [(0, 600, "<10m"), (600, 1800, "10-30m"), (1800, 3600, "30-60m"), (3600, 3 * 3600, "1-3h"),
            (3 * 3600, 6 * 3600, "3-6h"), (6 * 3600, 12 * 3600, "6-12h"), (12 * 3600, 86400, "12-24h"),
            (86400, 2 * 86400, "1-2d"), (2 * 86400, 3 * 86400, "2-3d"), (3 * 86400, 7 * 86400, "3-7d"),
            (7 * 86400, 14 * 86400, "7-14d"), (14 * 86400, 30 * 86400, "14-30d"), (30 * 86400, 90 * 86400, "30-90d"),
            (90 * 86400, 10**10, ">90d")]
GREC = [(0, 3600, "g<1h"), (3600, 86400, "g1-24h"), (86400, 7 * 86400, "g1-7d"), (7 * 86400, 10**10, "g>7d")]


def bins_case(expr, bins, other="NULL"):
    return "CASE " + " ".join(f"WHEN {expr} >= {a} AND {expr} < {b} THEN '{lab}'" for a, b, lab in bins) + f" ELSE {other} END"


def residualize(M, groups, tol=1e-10):
    """Remove multi-way fixed effects from every column of M (FWL).
    One grouping: exact within-group demeaning. Several: LSMR least squares on the sparse
    dummy design [G1 | G2 | ...] (exact up to tol); returns residuals and max LSMR iterations."""
    if len(groups) == 1:
        g = groups[0]
        cnt = np.bincount(g)
        R = M.copy()
        for j in range(M.shape[1]):
            R[:, j] -= (np.bincount(g, M[:, j]) / cnt)[g]
        return R, 1
    n = M.shape[0]
    blocks = [sparse.csr_matrix((np.ones(n), (np.arange(n), g)), shape=(n, g.max() + 1)) for g in groups]
    D = sparse.hstack(blocks).tocsr()
    R = np.empty_like(M)
    its = 0
    for j in range(M.shape[1]):
        sol = lsmr(D, M[:, j], atol=tol, btol=tol, maxiter=20000)
        R[:, j] = M[:, j] - D @ sol[0]
        its = max(its, sol[2])
    return R, its


def fe_ols(y, X, names, groups, cluster):
    keep = np.ones(len(y), bool)
    for g in groups:  # drop singletons (no within variation), iterate once more for safety
        keep &= np.bincount(g)[g] > 1
    y, X, cluster = y[keep], X[keep], cluster[keep]
    groups = [pd.factorize(g[keep])[0] for g in groups]
    Z, iters = residualize(np.column_stack([y, X]), groups)
    yt, Xt = Z[:, 0], Z[:, 1:]
    XtX = Xt.T @ Xt
    beta = np.linalg.solve(XtX, Xt.T @ yt)
    e = yt - Xt @ beta
    cl = pd.factorize(cluster)[0]
    S = np.zeros((Xt.shape[1], Xt.shape[1]))
    scores = np.zeros((cl.max() + 1, Xt.shape[1]))
    np.add.at(scores, cl, Xt * e[:, None])
    S = scores.T @ scores
    G, n, k = cl.max() + 1, len(yt), Xt.shape[1]
    inv = np.linalg.inv(XtX)
    V = inv @ S @ inv * G / (G - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(V))
    res = pd.DataFrame({"term": names, "coef": beta, "se": se, "ci_low": beta - 1.96 * se,
                        "ci_high": beta + 1.96 * se})
    res.attrs["V"] = V  # cluster-robust covariance (used for the RQ2 bootstrap of the causal reference)
    return res, dict(n=int(n), clusters=int(G), iters=iters)


def main(clean, export, out, test_period=False, window=("2026-01-01", "2026-04-01"), fine=False):
    lag_used = FINE_LAG if fine else LAG
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    c.execute(f"""CREATE TABLE ir AS
        SELECT r.uid, r.game_id, r.pos, r.game_time, r.cat_id * 10 + coalesce(l.ru, 0) AS pk, r.question_id, r.y::INT AS y,
               r.exposure, r.prev_y, r.gap_seconds, r.prior_games_all, r.high_volume::INT AS high_volume, r.history_complete,
               r.prior_games_all - lag(r.prior_games_all) OVER (PARTITION BY r.uid, r.question_id ORDER BY r.game_time, r.game_id) AS games_since
        FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id) WHERE r.game_time < TIMESTAMP '{END}'""")
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, pk, question_id FROM ir")
    # global recency of the drawn item: last deal in any earlier game
    c.execute("""CREATE TABLE grec AS SELECT i.uid, i.game_id, i.pos,
                 date_diff('second', d.game_time, i.game_time) AS g_rec
                 FROM ir i ASOF LEFT JOIN draws d ON d.pk = i.pk AND d.question_id = i.question_id AND d.game_time < i.game_time""")
    # leave-learner-out item difficulty on first exposures
    c.execute("""CREATE TABLE idiff AS SELECT question_id, sum(y) s, count(*) n FROM ir WHERE exposure = 1 GROUP BY 1""")
    c.execute("""CREATE TABLE udiff AS SELECT uid, question_id, sum(y) s, count(*) n FROM ir WHERE exposure = 1 GROUP BY ALL""")
    df = c.sql(f"""
        SELECT i.uid, i.game_id, CAST(i.game_time AS DATE) AS day, i.game_time, i.question_id, i.y, i.high_volume, i.history_complete,
               (i.exposure > 1)::INT AS seen, coalesce(i.prev_y, -1) AS prev_y,
               {bins_case('i.gap_seconds', lag_used)} AS lag_bin, i.gap_seconds,
               {bins_case('i.games_since', GAMES)} AS games_bin,
               {bins_case('g.g_rec', GREC, "'g_never'")} AS grec_bin,
               (d.s - coalesce(u.s, 0)) / nullif(d.n - coalesce(u.n, 0), 0) AS item_p_loo
        FROM ir i JOIN grec g USING (uid, game_id, pos)
        JOIN idiff d USING (question_id) LEFT JOIN udiff u USING (uid, question_id)""").df()
    if test_period:
        df = df[(df.game_time >= pd.Timestamp(window[0])) & (df.game_time < pd.Timestamp(window[1]))].reset_index(drop=True)
    print("responses:", len(df), "learners:", df.uid.nunique(), "seen share:", round(df.seen.mean(), 4), flush=True)

    y = df.y.values.astype(float)
    g_game = pd.factorize(df.uid.astype(str) + "_" + df.game_id.astype(str))[0]
    g_day = pd.factorize(df.uid.astype(str) + "_" + df.day.astype(str))[0]
    g_item = pd.factorize(df.question_id)[0]
    cluster = df.uid.values

    def dummies(col, levels, mask=None):
        cols, names = [], []
        for lev in levels:
            v = (df[col].values == lev)
            if mask is not None:
                v = v & mask
            cols.append(v.astype(float)); names.append(lev)
        return cols, names

    gcols, gnames = dummies("grec_bin", ["g<1h", "g1-24h", "g1-7d", "g_never"])  # ref: g>7d
    lag_levels = [lab for _, _, lab in lag_used]
    game_levels = [lab for _, _, lab in GAMES]
    specs = {}
    specs["A seen"] = ([df.seen.values.astype(float)], ["seen"])
    lc, ln = dummies("lag_bin", lag_levels)
    specs["B lag"] = (lc, ["lag " + n for n in ln])
    c1, n1 = dummies("lag_bin", lag_levels, df.prev_y.values == 1)
    c0, n0 = dummies("lag_bin", lag_levels, df.prev_y.values == 0)
    specs["C lag x prev"] = (c1 + c0, ["after correct, lag " + n for n in n1] + ["after error, lag " + n for n in n0])
    gc, gn = dummies("games_bin", game_levels)
    specs["D games"] = (gc, ["games since " + n for n in gn])

    if test_period:
        specs = {k: v for k, v in specs.items() if not k.startswith("D")}
    if fine:
        specs = {k: v for k, v in specs.items() if k.startswith(("B", "C"))}
    estimators = {
        "E1 learner-game FE + item FE": ([g_game, g_item], None),
        "E2 learner-day FE + item FE": ([g_day, g_item], None),
        "E3 learner-game FE + item difficulty": ([g_game], "item_p_loo"),
    }
    if test_period or fine:
        estimators = {k: v for k, v in estimators.items() if k.startswith("E1")}
    sfx = ("_test" + ("" if window[0] == "2026-01-01" else "_" + window[0][:7])) if test_period else ("_fine" if fine else "")
    rows, info = [], []
    for sname, (cols, names) in specs.items():
        for ename, (groups, extra) in estimators.items():
            X = np.column_stack(cols + gcols + ([np.nan_to_num(df[extra].values, nan=np.nanmean(df[extra].values))] if extra else []))
            nm = names + ["ctrl " + g for g in gnames] + ([extra] if extra else [])
            res, meta = fe_ols(y, X, nm, groups, cluster)
            if test_period or fine:  # covariance of the causal reference (RQ2 bootstrap, RQ3 policy draws)
                np.save(os.path.join(out, f"rq1_cov{sfx}_{sname.split()[0]}.npy"), res.attrs["V"])
                pd.Series(nm).to_csv(os.path.join(out, f"rq1_cov{sfx}_{sname.split()[0]}_terms.csv"), index=False)
            res.attrs.clear()  # pandas 3 compares attrs in concat; the covariance array must not travel with the frame
            res.insert(0, "estimator", ename); res.insert(0, "spec", sname)
            rows.append(res)
            info.append(dict(spec=sname, estimator=ename, **meta))
            print(sname, "|", ename, meta, flush=True)
    out_df = pd.concat(rows, ignore_index=True)
    out_df.to_csv(os.path.join(out, f"rq1_effects{sfx}.csv"), index=False, float_format="%.5f")
    pd.DataFrame(info).to_csv(os.path.join(out, f"rq1_model_info{sfx}.csv"), index=False)

    # descriptive anchors
    base = df[df.seen == 0].y.mean()
    desc = df.groupby(["seen"]).y.agg(["mean", "count"]).reset_index()
    desc.to_csv(os.path.join(out, f"rq1_descriptive{sfx}.csv"), index=False, float_format="%.4f")

    if fine:  # representative (geometric-mean) lag of each fine bin among re-exposures
        sel = df[df.seen == 1]
        rep = sel.groupby("lag_bin").gap_seconds.apply(lambda v: float(np.exp(np.log(np.maximum(v / 3600.0, 1e-3)).mean())))
        rep.reindex(lag_levels).rename("hours").to_csv(os.path.join(out, "rq1_fine_lag_hours.csv"))
    if test_period or fine:
        pd.set_option("display.width", 220)
        print("P(correct | unseen) =", round(base, 4))
        show = out_df[~out_df.term.str.startswith("ctrl")]
        print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return
    # robustness of A/E1 on subgroups
    rob = []
    for lab, m in [("not high_volume", df.high_volume.values == 0), ("cohort 2025+", df.history_complete.values == 1)]:
        dd = m
        res, meta = fe_ols(y[dd], np.column_stack([df.seen.values[dd].astype(float)] + [c_[dd] for c_ in gcols]),
                           ["seen"] + ["ctrl " + g for g in gnames],
                           [pd.factorize(g_game[dd])[0], pd.factorize(g_item[dd])[0]], cluster[dd])
        r = res.iloc[0].to_dict(); r.update(subset=lab, **meta); rob.append(r)
    pd.DataFrame(rob).to_csv(os.path.join(out, "rq1_robustness.csv"), index=False, float_format="%.5f")

    pd.set_option("display.width", 220)
    print("\nP(correct | unseen) =", round(base, 4))
    show = out_df[~out_df.term.str.startswith("ctrl") & (out_df.term != "item_p_loo")]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(pd.DataFrame(rob).to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    a = sys.argv
    win = tuple(a[a.index("--window") + 1:a.index("--window") + 3]) if "--window" in a else ("2026-01-01", "2026-04-01")
    main(a[1], a[2], a[3], test_period="--test-period" in a[4:], window=win, fine="--fine-lags" in a[4:])
