"""
RQ1 robustness: conditional logit within learner-game groups (non-linear counterpart of E1/E3).

    python rq1d_clogit.py <clean_dir> <export_dir> <out_dir> [--window START END]

--window: estimation sample restricted to a test window (histories and draws still use all earlier data);
adds specification C (lag bins x previous outcome) and saves the cluster-robust covariances
rq1d_cov_<sfx>_<B|C>.npy (+ _terms.csv) for the log-odds consistency analysis of RQ2. Files *_test[_yyyy-mm].

Conditioning on the number of correct answers among a learner's round-5 items in a game removes the
learner-game effect exactly (Chamberlain conditional likelihood), without the LPM's additivity at the
edges of the probability scale. Groups have 2-3 items; groups with all-correct or all-wrong answers carry
no information and drop out. Regressors: seen (spec A) or lag bins (spec B), logit of the item's
leave-learner-out first-exposure p-correct (item difficulty, as in E3) and the global-recency controls.
Learner-clustered sandwich SE. Effects reported as odds ratios and as percentage points at the baseline
P(correct | unseen) = 0.54 (expit(logit(0.54) + beta) - 0.54), for comparison with the LPM estimates.
Period: games before 2026-04-01.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit, logsumexp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq1_causal import GREC, LAG, bins_case  # noqa: E402

END = "2026-04-01"


def clogit(y, X, grp):
    """Conditional logit for groups of size 2-3. Returns beta, cluster-robust V needs per-row scores."""
    order = np.argsort(grp, kind="stable")
    y, X, grp = y[order], X[order], grp[order]
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    sizes = np.diff(np.r_[starts, len(grp)])
    ksum = np.add.reduceat(y, starts)
    keep_g = (ksum > 0) & (ksum < sizes) & (sizes <= 3)
    row_keep = np.repeat(keep_g, sizes)
    y, X, grp = y[row_keep], X[row_keep], grp[row_keep]
    starts = np.flatnonzero(np.r_[True, grp[1:] != grp[:-1]])
    sizes = np.diff(np.r_[starts, len(grp)])
    ksum = np.add.reduceat(y, starts)
    gid = np.repeat(np.arange(len(starts)), sizes)
    # pad to (G, 3)
    G = len(starts)
    pos = np.arange(len(grp)) - np.repeat(starts, sizes)
    Xg = np.zeros((G, 3, X.shape[1])); Mg = np.zeros((G, 3), bool); Yg = np.zeros((G, 3))
    Xg[gid, pos], Mg[gid, pos], Yg[gid, pos] = X, True, y
    use_comp = (ksum == sizes - 1) & (sizes == 3)  # k = 2 of 3: denominator over complements

    def parts(beta):
        eta = np.where(Mg, Xg @ beta, -np.inf)
        S = np.where(Mg, eta, 0).sum(1)
        num = (np.where(Mg, eta, 0) * Yg).sum(1)
        alt = np.where(use_comp[:, None], np.where(Mg, S[:, None] - eta, -np.inf), eta)  # log terms of the denominator
        den = logsumexp(alt, axis=1)
        w = np.exp(alt - den[:, None])  # weight of each denominator term
        # expected x under the conditional distribution
        xs = Xg.sum(1)
        ex = np.where(use_comp[:, None], xs - np.einsum("gk,gkp->gp", w, Xg), np.einsum("gk,gkp->gp", w, Xg))
        xobs = np.einsum("gk,gkp->gp", Yg, Xg)
        return num - den, xobs - ex  # per-group log-lik and score

    def nll(beta):
        ll, sc = parts(beta)
        return -ll.sum(), -sc.sum(0)
    res = minimize(nll, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options=dict(maxiter=500, gtol=1e-8))
    beta = res.x
    # numerical Hessian from analytic scores
    eps = 1e-5
    H = np.zeros((len(beta), len(beta)))
    for j in range(len(beta)):
        d = np.zeros_like(beta); d[j] = eps
        H[:, j] = (nll(beta + d)[1] - nll(beta - d)[1]) / (2 * eps)
    _, sc = parts(beta)
    return beta, H, sc, starts, order, row_keep, res


def main(clean, export, out, window=None):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    c.execute(f"""CREATE TABLE ir AS SELECT r.uid, r.game_id, r.pos, r.game_time, r.cat_id * 10 + coalesce(l.ru, 0) AS pk,
                  r.question_id, r.y::INT AS y, r.exposure, r.gap_seconds, r.prev_y
                  FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id) WHERE r.game_time < TIMESTAMP '{END}'""")
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, pk, question_id FROM ir")
    c.execute("""CREATE TABLE grec AS SELECT i.uid, i.game_id, i.pos, date_diff('second', d.game_time, i.game_time) AS g_rec
                 FROM ir i ASOF LEFT JOIN draws d ON d.pk = i.pk AND d.question_id = i.question_id AND d.game_time < i.game_time""")
    c.execute("CREATE TABLE idiff AS SELECT question_id, sum(y) s, count(*) n FROM ir WHERE exposure = 1 GROUP BY 1")
    c.execute("CREATE TABLE udiff AS SELECT uid, question_id, sum(y) s, count(*) n FROM ir WHERE exposure = 1 GROUP BY ALL")
    df = c.sql(f"""
        SELECT i.uid, i.game_id, i.game_time, i.y, (i.exposure > 1)::INT AS seen, coalesce(i.prev_y, -1) AS prev_y,
               {bins_case('i.gap_seconds', LAG)} AS lag_bin,
               {bins_case('g.g_rec', GREC, "'g_never'")} AS grec_bin,
               (d.s - coalesce(u.s, 0) + 1) / (d.n - coalesce(u.n, 0) + 2) AS item_p
        FROM ir i JOIN grec g USING (uid, game_id, pos) JOIN idiff d USING (question_id) LEFT JOIN udiff u USING (uid, question_id)""").df()
    if window:
        df = df[(df.game_time >= pd.Timestamp(window[0])) & (df.game_time < pd.Timestamp(window[1]))].reset_index(drop=True)
    sfx = ("_test" + ("" if window[0] == "2026-01-01" else "_" + window[0][:7])) if window else ""
    print("responses:", len(df), flush=True)
    grp = pd.factorize(df.uid.astype(str) + "_" + df.game_id.astype(str))[0]
    y = df.y.values.astype(float)
    ctrl = np.column_stack([logit(df.item_p.values)] + [(df.grec_bin.values == g).astype(float) for g in ["g<1h", "g1-24h", "g1-7d", "g_never"]])
    cnames = ["logit item p", "ctrl g<1h", "ctrl g1-24h", "ctrl g1-7d", "ctrl g_never"]
    lags = [lab for _, _, lab in LAG]
    specs = {"A seen": (df.seen.values[:, None].astype(float), ["seen"]),
             "B lag": (np.column_stack([(df.lag_bin.values == l).astype(float) for l in lags]), ["lag " + l for l in lags])}
    if window:
        pv = df.prev_y.values
        specs["C lag x prev"] = (np.column_stack([((df.lag_bin.values == l) & (pv == 1)).astype(float) for l in lags] +
                                                 [((df.lag_bin.values == l) & (pv == 0)).astype(float) for l in lags]),
                                 ["after correct, lag " + l for l in lags] + ["after error, lag " + l for l in lags])
    base_p = 0.54
    rows = []
    for sname, (Xs, names) in specs.items():
        X = np.column_stack([Xs, ctrl])
        beta, H, sc, starts, order, row_keep, res = clogit(y, X, grp)
        uid_sorted = df.uid.values[order][row_keep]
        g_uid = uid_sorted[starts]
        cl = pd.factorize(g_uid)[0]
        Sc = np.zeros((cl.max() + 1, len(beta)))
        np.add.at(Sc, cl, sc)
        Hi = np.linalg.inv(H)
        V = Hi @ (Sc.T @ Sc) @ Hi
        se = np.sqrt(np.diag(V))
        print(sname, "groups used:", len(starts), "converged:", res.success, flush=True)
        if window:
            np.save(os.path.join(out, f"rq1d_cov{sfx}_{sname.split()[0]}.npy"), V)
            pd.Series(names + cnames).to_csv(os.path.join(out, f"rq1d_cov{sfx}_{sname.split()[0]}_terms.csv"), index=False)
        for k, nm in enumerate(names + cnames):
            rows.append(dict(spec=sname, term=nm, beta=beta[k], se=se[k], odds_ratio=np.exp(beta[k]),
                             pp_at_054=(expit(logit(base_p) + beta[k]) - base_p) * 100,
                             pp_lo=(expit(logit(base_p) + beta[k] - 1.96 * se[k]) - base_p) * 100,
                             pp_hi=(expit(logit(base_p) + beta[k] + 1.96 * se[k]) - base_p) * 100,
                             groups=len(starts)))
    res_df = pd.DataFrame(rows)
    res_df.to_csv(os.path.join(out, f"rq1d_clogit{sfx}.csv"), index=False, float_format="%.5f")
    pd.set_option("display.width", 200)
    print(res_df[~res_df.term.str.startswith("ctrl")].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], a[2], a[3], window=tuple(a[a.index("--window") + 1:a.index("--window") + 3]) if "--window" in a else None)
