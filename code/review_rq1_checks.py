"""
Reviewer checks of the RQ1 causal estimates (identification, measurement, informative timing, inference).

    python review_rq1_checks.py <clean_dir> <export_dir> <out_dir>

Period: games before 2026-04-01; item level = round 5; estimator E1 (learner x game FE + item FE, global-recency
controls) exactly as in rq1_causal.py, whose helpers are reused.

(1) Unrecorded encounters in rounds 1-4 (item ids are stored for round 5 only). Rounds 1-4 draw items from the same
    category-language pools at random, so the expected number of unrecorded encounters of learner u with item i
    before time t is
        lambda_ui(t) = sum over u's round-1..4 rounds r in pool(i) before t of 3 * f_i(month(r)),
    f_i = share of item i among round-5 draws of its pool in that month (each round deals 3 items).
    For a re-exposure, lambda_int = lambda(t) - lambda(t_prev) counts unrecorded encounters after the last recorded one.
    Outputs: size of the problem; E1 with lambda as a covariate (its coefficient = effect of an expected unrecorded
    encounter); E1 on a 'clean' subsample with P(unrecorded encounter) < 0.05.
(2) Informative return time: lag curve identified only from comparisons between previously seen items WITHIN the same
    game (return time held fixed; unseen items dropped; reference = >90 d); curves by learner activity tercile,
    by cohort and among learners still active 60+ days after the game (games before 2026-01-30).
(3) Dose / history: seen split by the number of recorded earlier encounters; seen x log(prior games).
(4) Inference: two-way cluster-robust SE (learner and item; Cameron-Gelbach-Miller) for specifications A and B.
Only aggregates are written.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

from rq1_causal import END, GREC, LAG, bins_case, residualize

ACT_END = "2026-04-01"


def fe_ols2(y, X, names, groups, cl1, cl2):
    """E1-type FE regression with one-way (cl1) and two-way (cl1, cl2) cluster-robust SE."""
    keep = np.ones(len(y), bool)
    for g in groups:
        keep &= np.bincount(g)[g] > 1
    y, X, cl1, cl2 = y[keep], X[keep], cl1[keep], cl2[keep]
    groups = [pd.factorize(g[keep])[0] for g in groups]
    Z, iters = residualize(np.column_stack([y, X]), groups)
    yt, Xt = Z[:, 0], Z[:, 1:]
    ok = Xt.std(axis=0) > 1e-9  # columns without variation after absorbing the fixed effects are dropped
    Xt, names = Xt[:, ok], [nm for nm, k_ in zip(names, ok) if k_]
    XtX = Xt.T @ Xt
    inv = np.linalg.inv(XtX)
    beta = inv @ (Xt.T @ yt)
    e = yt - Xt @ beta
    n, k = len(yt), Xt.shape[1]

    def meat(cl):
        c = pd.factorize(cl)[0]
        sc = np.zeros((c.max() + 1, k))
        np.add.at(sc, c, Xt * e[:, None])
        G = c.max() + 1
        return sc.T @ sc * G / (G - 1) * (n - 1) / (n - k), G
    S1, G1 = meat(cl1)
    S2, G2 = meat(cl2)
    S12, _ = meat(pd.Series(cl1).astype(str).values + "_" + pd.Series(cl2).astype(str).values)
    V1 = inv @ S1 @ inv
    V2 = inv @ (S1 + S2 - S12) @ inv
    d2 = np.diag(V2)
    if (d2 <= 0).any():  # CGM can be non-PSD; fall back to the larger one-way variance for that term
        d2 = np.where(d2 > 0, d2, np.maximum(np.diag(V1), np.diag(inv @ S2 @ inv)))
    return pd.DataFrame({"term": names, "coef": beta, "se_learner": np.sqrt(np.diag(V1)), "se_twoway": np.sqrt(d2)}), \
        dict(n=int(n), learners=int(G1), items=int(G2), iters=iters)


def main(clean, export, out):
    os.makedirs(out, exist_ok=True)
    tmp = os.path.join(out, "_duckdb_tmp")
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute(f"SET temp_directory='{tmp}'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    c.execute(f"""CREATE TABLE ir AS
        SELECT row_number() OVER (ORDER BY r.uid, r.game_id, r.pos) AS rid,
               r.uid, r.game_id, r.pos, r.game_time, r.cat_id * 10 + coalesce(l.ru, 0) AS pk, r.question_id, r.y::INT AS y,
               r.exposure, r.prev_y, r.gap_seconds, r.prior_games_all, r.history_complete
        FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id) WHERE r.game_time < TIMESTAMP '{END}'""")
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, pk, question_id FROM ir")
    c.execute("""CREATE TABLE grec AS SELECT i.rid, date_diff('second', d.game_time, i.game_time) AS g_rec
                 FROM ir i ASOF LEFT JOIN draws d ON d.pk = i.pk AND d.question_id = i.question_id AND d.game_time < i.game_time""")

    # ---------------- (1) expected unrecorded encounters in rounds 1-4 ----------------
    c.execute("CREATE TABLE glang AS SELECT game_id, round(avg(pk % 10))::INT AS ru FROM ir GROUP BY 1")
    c.execute(f"""CREATE TABLE rr AS SELECT DISTINCT k.uid, k.game_id, k.round, k.cat_id * 10 + g.ru AS pk, k.game_time,
                  date_trunc('month', k.game_time) AS m
                  FROM {P('kc_responses')} k JOIN glang g USING (game_id) WHERE k.round < 5 AND k.game_time < TIMESTAMP '{END}'""")
    c.execute("""CREATE TABLE rcum AS SELECT uid, pk, m, game_time,
                 row_number() OVER (PARTITION BY uid, pk, m ORDER BY game_time, game_id, round) AS cum FROM rr""")
    c.execute("CREATE TABLE rcum_last AS SELECT uid, pk, m, game_time, max(cum) AS cum FROM rcum GROUP BY ALL")  # rounds up to t
    c.execute("CREATE TABLE nm AS SELECT uid, pk, m, count(*) AS n FROM rr GROUP BY ALL")
    c.execute("""CREATE TABLE fshare AS WITH a AS (SELECT pk, date_trunc('month', game_time) m, question_id, count(*) n FROM draws GROUP BY ALL)
                 SELECT pk, m, question_id, n / sum(n) OVER (PARTITION BY pk, m) AS f FROM a""")
    # query points: each response at t, and each re-exposure at the time of the previous recorded encounter
    c.execute("""CREATE TABLE q AS
        SELECT rid, 0 AS at_prev, uid, pk, question_id, game_time AS t FROM ir
        UNION ALL
        SELECT rid, 1, uid, pk, question_id, game_time - to_seconds(gap_seconds) FROM ir WHERE exposure > 1""")
    c.execute("""CREATE TABLE lam_full AS
        SELECT q.rid, q.at_prev, sum(n.n * 3 * coalesce(f.f, 0)) AS lam
        FROM q JOIN nm n ON n.uid = q.uid AND n.pk = q.pk AND n.m < date_trunc('month', q.t)
        LEFT JOIN fshare f ON f.pk = q.pk AND f.m = n.m AND f.question_id = q.question_id
        GROUP BY ALL""")
    c.execute("""CREATE TABLE lam_part AS
        SELECT q.rid, q.at_prev,
               CASE WHEN r.m = date_trunc('month', q.t) THEN r.cum ELSE 0 END * 3 * coalesce(f.f, 0) AS lam
        FROM q ASOF LEFT JOIN rcum_last r ON r.uid = q.uid AND r.pk = q.pk AND q.t >= r.game_time
        LEFT JOIN fshare f ON f.pk = q.pk AND f.m = date_trunc('month', q.t) AND f.question_id = q.question_id""")
    c.execute("""CREATE TABLE lam AS SELECT rid, at_prev, sum(lam) AS lam FROM
                 (SELECT * FROM lam_full UNION ALL SELECT * FROM lam_part) GROUP BY ALL""")
    print("lambda computed", flush=True)

    # learner activity (games per active day, analysis period) and last active day
    c.execute(f"""CREATE TABLE act AS SELECT uid, sum(games) / count(*) AS games_per_day, max(d) AS last_day
                  FROM {P('activity_days')} WHERE d < DATE '{ACT_END}' GROUP BY 1""")
    df = c.sql(f"""
        SELECT i.rid, i.uid, i.game_id, i.game_time, i.question_id, i.y, i.history_complete, i.prior_games_all,
               (i.exposure > 1)::INT AS seen, i.exposure, coalesce(i.prev_y, -1) AS prev_y, i.gap_seconds,
               {bins_case('i.gap_seconds', LAG)} AS lag_bin,
               {bins_case('g.g_rec', GREC, "'g_never'")} AS grec_bin,
               coalesce(l0.lam, 0) AS lam_t, coalesce(l1.lam, 0) AS lam_prev,
               a.games_per_day, a.last_day
        FROM ir i JOIN grec g USING (rid)
        LEFT JOIN lam l0 ON l0.rid = i.rid AND l0.at_prev = 0
        LEFT JOIN lam l1 ON l1.rid = i.rid AND l1.at_prev = 1
        LEFT JOIN act a USING (uid)""").df()
    df["lam_int"] = np.where(df.seen == 1, np.maximum(df.lam_t - df.lam_prev, 0), np.nan)
    df["p_unrec_before"] = 1 - np.exp(-df.lam_t)            # any unrecorded encounter before t
    df["p_unrec_int"] = 1 - np.exp(-df.lam_int)             # after the last recorded one (re-exposures)
    print("responses:", len(df), flush=True)

    # descriptives of the measurement problem
    unseen, seen = df.seen == 0, df.seen == 1
    desc = [dict(group="unseen (control)", n=int(unseen.sum()), mean_lambda=df.lam_t[unseen].mean(),
                 mean_p_unrecorded=df.p_unrec_before[unseen].mean(),
                 share_p_gt_0_05=(df.p_unrec_before[unseen] > 0.05).mean(), share_p_gt_0_25=(df.p_unrec_before[unseen] > 0.25).mean())]
    for lab, g in df[seen].groupby("lag_bin", sort=False):
        desc.append(dict(group=f"re-exposure, lag {lab}", n=len(g), mean_lambda=g.lam_int.mean(),
                         mean_p_unrecorded=g.p_unrec_int.mean(), share_p_gt_0_05=(g.p_unrec_int > 0.05).mean(),
                         share_p_gt_0_25=(g.p_unrec_int > 0.25).mean()))
    desc.append(dict(group="re-exposure, all", n=int(seen.sum()), mean_lambda=df.lam_int[seen].mean(),
                     mean_p_unrecorded=df.p_unrec_int[seen].mean(), share_p_gt_0_05=(df.p_unrec_int[seen] > 0.05).mean(),
                     share_p_gt_0_25=(df.p_unrec_int[seen] > 0.25).mean()))
    pd.DataFrame(desc).to_csv(os.path.join(out, "unrecorded_exposure_descriptives.csv"), index=False, float_format="%.5f")
    print(pd.DataFrame(desc).to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    lag_levels = [lab for _, _, lab in LAG]
    gnames = ["g<1h", "g1-24h", "g1-7d", "g_never"]

    def design(d, spec, extra=()):
        cols, names = [], []
        if spec == "A":
            cols.append(d.seen.values.astype(float)); names.append("seen")
        elif spec in ("B", "Bseen"):
            levels = lag_levels if spec == "B" else lag_levels[:-1]  # Bseen: reference = >90d
            for lev in levels:
                cols.append((d.lag_bin.values == lev).astype(float)); names.append("lag " + lev)
        elif spec == "dose":
            k = d.exposure.values - 1
            for lab, m in (("1 earlier", k == 1), ("2 earlier", k == 2), ("3+ earlier", k >= 3)):
                cols.append(m.astype(float)); names.append("seen, " + lab)
        for e in extra:
            if e == "lam_t":
                cols.append(d.lam_t.values); names.append("unrecorded encounters (expected)")
            if e == "seen_x_logexp":
                cols.append(d.seen.values * np.log1p(d.prior_games_all.values)); names.append("seen x log(1+prior games)")
        for gname in gnames:
            cols.append((d.grec_bin.values == gname).astype(float)); names.append("ctrl " + gname)
        return np.column_stack(cols), names

    rows, info = [], []
    done = set()
    if "--resume" in sys.argv:
        prev = os.path.join(out, "review_rq1_effects.csv")
        if os.path.exists(prev):
            done = set(pd.read_csv(prev).model)

    def fit(label, d, spec, extra=()):
        if label in done:
            print("skip (done):", label, flush=True)
            return
        X, names = design(d, spec, extra)
        groups = [pd.factorize(d.uid.astype(str) + "_" + d.game_id.astype(str))[0], pd.factorize(d.question_id)[0]]
        res, meta = fe_ols2(d.y.values.astype(float), X, names, groups, d.uid.values, d.question_id.values)
        res = res[~res.term.str.startswith("ctrl")]
        res.insert(0, "model", label)
        rows.append(res); info.append(dict(model=label, **meta))
        print(label, meta, flush=True)
        path = os.path.join(out, "review_rq1_effects.csv")  # append after every fit (long run)
        res.to_csv(path, mode="a", header=not os.path.exists(path), index=False, float_format="%.5f")
        print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"), flush=True)

    # (4) baseline with two-way SE  +  (1) unrecorded encounters
    fit("A baseline", df, "A")
    fit("A + expected unrecorded encounters", df, "A", ("lam_t",))
    clean_ = ((df.seen == 0) & (df.p_unrec_before < 0.05)) | ((df.seen == 1) & (df.p_unrec_int < 0.05))
    fit("A clean (P unrecorded < 0.05)", df[clean_].reset_index(drop=True), "A")
    fit("B baseline", df, "B")
    fit("B + expected unrecorded encounters", df, "B", ("lam_t",))
    fit("B clean (P unrecorded < 0.05)", df[clean_].reset_index(drop=True), "B")
    # (3) dose and history
    fit("A dose (number of earlier recorded encounters)", df, "dose")
    fit("A + seen x log(prior games)", df, "A", ("seen_x_logexp",))
    # (2) informative return time
    fit("B within-game, seen items only (ref >90d)", df[df.seen == 1].reset_index(drop=True), "Bseen")
    # same learners, ordinary specification B (all items), for a like-for-like comparison with the within-game curve
    sd = df[df.seen == 1]
    multi = sd.groupby(["uid", "game_id"]).lag_bin.transform("nunique") > 1
    wg_users = set(sd.loc[multi, "uid"])
    fit("B learners of the within-game sample, all items", df[df.uid.isin(wg_users)].reset_index(drop=True), "B")
    terc = pd.qcut(df.groupby("uid").games_per_day.first(), 3, labels=["low", "mid", "high"])
    df["act_terc"] = df.uid.map(terc).astype(str)
    for t in ("low", "mid", "high"):
        fit(f"B activity tercile {t}", df[df.act_terc == t].reset_index(drop=True), "B")
    for h, lab in ((1, "cohort 2025+ (complete history)"), (0, "registered before 2025")):
        fit(f"B {lab}", df[df.history_complete == h].reset_index(drop=True), "B")
    surv = (df.game_time < pd.Timestamp("2026-01-30")) & (pd.to_datetime(df.last_day) >= df.game_time.dt.normalize() + pd.Timedelta(days=60))
    early = df.game_time < pd.Timestamp("2026-01-30")
    fit("B games before 2026-01-30, all learners", df[early].reset_index(drop=True), "B")
    fit("B games before 2026-01-30, learners active 60+ days later", df[surv].reset_index(drop=True), "B")

    pd.DataFrame(info).to_csv(os.path.join(out, "review_rq1_info.csv"), index=False)
    # lag distribution of re-exposures by activity tercile (descriptive: return time is informative)
    lagd = df[df.seen == 1].groupby("act_terc").gap_seconds.describe(percentiles=[0.25, 0.5, 0.75]) / 3600
    lagd.to_csv(os.path.join(out, "lag_hours_by_activity.csv"), float_format="%.2f")
    print(lagd.to_string(float_format=lambda v: f"{v:.1f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
