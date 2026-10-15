"""
Is the suppression of short repeats a property of the learner or of a global "deck"?

    python randomization_deck.py <clean_dir> <export_dir> <out_dir> [sample_frac]

Developers state that items are drawn at random, without regard to the player and without an
anti-repeat rule. A shuffled global deck dealt without replacement (e.g. a cached queue) would
still make an item that was just dealt to ANYONE less likely to be dealt again soon.

For every sampled response and every item of its pool (category x item language, month; weights
leave-learner-out, as in randomization_check.py) we record
    global recency = time since the item was last dealt in any other game (before this game)
    learner status = item unseen by the learner / seen by the learner before
and compare observed with expected draws in each cell. Under independent draws the ratio is 1 in
every global-recency bin (time since the last draw carries no information given the draw share).
Deck: ratio < 1 at short global recency, for unseen items too. If, at equal global recency, seen
and unseen items have the same ratio, the learner-level suppression is fully explained by the deck.
Period: games before 2026-04-01. Only aggregates are written.
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

SEED = 42
N_BOOT = 300
END = "2026-04-01"
R_BINS = [(0, 600, "<10min"), (600, 3600, "10-60min"), (3600, 6 * 3600, "1-6h"),
          (6 * 3600, 86400, "6-24h"), (86400, 7 * 86400, "1-7d"), (7 * 86400, 10**10, ">7d")]


def main(clean, export, out, frac=0.02):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    c.execute("SET memory_limit='12GB'")
    c.execute(f"SET temp_directory='{os.path.join(out, '_duckdb_tmp')}'")
    P = lambda n: f"'{os.path.join(clean, n + '.parquet')}'"
    ns = chr(92) + "N"
    c.execute(f"""CREATE TABLE qlang AS SELECT id AS question_id, (lang = 'ru')::INT AS ru FROM
                  read_csv('{os.path.join(export, 'questions.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1)""")
    c.execute(f"""CREATE TABLE ir AS SELECT r.uid, r.game_id, r.game_time, r.cat_id * 10 + coalesce(l.ru, 0) AS pk, r.question_id
                  FROM {P('item_responses')} r LEFT JOIN qlang l USING (question_id) WHERE r.game_time < TIMESTAMP '{END}'""")
    c.execute("CREATE TABLE draws AS SELECT DISTINCT game_id, game_time, pk, question_id FROM ir")
    c.execute("""CREATE TABLE pool AS SELECT pk, date_trunc('month', game_time) w, question_id, count(*) n FROM draws GROUP BY ALL""")
    c.execute("CREATE TABLE pool_tot AS SELECT pk, w, sum(n) n_tot FROM pool GROUP BY ALL")
    c.execute("""CREATE TABLE own AS SELECT r.uid, d.pk, date_trunc('month', d.game_time) w, d.question_id, count(*) n_own
                 FROM draws d JOIN (SELECT DISTINCT uid, game_id FROM ir) r USING (game_id) GROUP BY ALL""")
    c.execute("CREATE TABLE own_tot AS SELECT uid, pk, w, sum(n_own) n_own_tot FROM own GROUP BY ALL")
    c.execute(f"""CREATE TABLE s AS SELECT row_number() OVER () rid, uid, game_id, game_time, pk, question_id AS drawn,
                  date_trunc('month', game_time) w FROM ir
                  WHERE hash(uid, game_id, question_id, {SEED}) % 10000 < {int(frac * 10000)}""")
    c.execute("""CREATE TABLE seen AS SELECT DISTINCT s.rid, e.question_id FROM s
                 JOIN ir e ON e.uid = s.uid AND e.pk = s.pk AND e.game_time < s.game_time""")
    # candidates: every pool item of the response's pool-month, with leave-learner-out weight
    c.execute("""CREATE TABLE cand AS
        SELECT s.rid, s.uid, s.game_id, s.game_time, s.pk, p.question_id,
               (p.question_id = s.drawn)::INT AS obs,
               (p.n - coalesce(o.n_own, 0)) / (t.n_tot - ot.n_own_tot) AS e,
               (sn.question_id IS NOT NULL)::INT AS seen_by_learner
        FROM s JOIN pool p ON p.pk = s.pk AND p.w = s.w
        JOIN pool_tot t ON t.pk = s.pk AND t.w = s.w
        JOIN own_tot ot ON ot.uid = s.uid AND ot.pk = s.pk AND ot.w = s.w
        LEFT JOIN own o ON o.uid = s.uid AND o.pk = s.pk AND o.w = s.w AND o.question_id = p.question_id
        LEFT JOIN seen sn ON sn.rid = s.rid AND sn.question_id = p.question_id
        WHERE t.n_tot - ot.n_own_tot > 0""")
    print("responses:", c.execute("SELECT count(*) FROM s").fetchone()[0],
          "candidate rows:", c.execute("SELECT count(*) FROM cand").fetchone()[0], flush=True)
    # global recency: last time the item was dealt in any game strictly before this game
    c.execute("""CREATE TABLE cand_r AS
        SELECT c.*, date_diff('second', d.game_time, c.game_time) AS rec
        FROM cand c ASOF LEFT JOIN draws d
          ON d.pk = c.pk AND d.question_id = c.question_id AND d.game_time < c.game_time""")
    rcase = "CASE " + " ".join(f"WHEN rec >= {a} AND rec < {b} THEN '{lab}'" for a, b, lab in R_BINS) + " ELSE 'never' END"
    cells = c.sql(f"""SELECT uid, seen_by_learner, {rcase} rb, sum(obs) o, sum(e) e FROM cand_r GROUP BY ALL""").df()

    rng = np.random.default_rng(SEED)
    rows = []
    for (sb, rb), g in cells.groupby(["seen_by_learner", "rb"]):
        o, e = g.o.values, g.e.values
        boots = [o[k].sum() / e[k].sum() for k in (rng.integers(0, len(g), len(g)) for _ in range(N_BOOT))]
        rows.append(dict(learner_status="seen" if sb else "unseen", global_recency=rb, observed=int(o.sum()),
                         expected=round(e.sum(), 1), ratio=o.sum() / e.sum(),
                         ci_low=np.percentile(boots, 2.5), ci_high=np.percentile(boots, 97.5)))
    res = pd.DataFrame(rows)
    order = {lab: i for i, (_, _, lab) in enumerate(R_BINS)} | {"never": 99}
    res = res.sort_values(["learner_status", "global_recency"], key=lambda s: s.map(order) if s.name == "global_recency" else s)
    res.to_csv(os.path.join(out, "deck_global_recency.csv"), index=False, float_format="%.4f")
    pd.set_option("display.width", 200)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.02)
