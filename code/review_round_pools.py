"""
Do rounds 1-4 draw items from the same pools as round 5? (reviewer comment on unrecorded encounters)

    python review_round_pools.py <export_dir> <out_dir>

Item ids are stored only for round 5 (games.questions), but the application keeps lifetime answer counters per item
(questions.fail + questions.correct, all rounds and modes). For items created on or after 2025-01-01 the whole lifetime
lies inside the export, so the counter can be compared with the item's round-5 deals in all exported games
(two players answer every deal):
    ratio_i = (fail_i + correct_i) / (2 * round-5 deals_i).
If items were drawn at random from one pool in every round, each category c would add R_c = (rounds 1-4 played in c) /
(round-5 rounds played in c) answers per round-5 answer, so ratio ~ 1 + R_c (about 5-8 here) and increasing in R_c
with slope 1. If round 5 uses its own items, the ratio does not depend on R_c. Only aggregates are written.
"""
import os
import sys

import duckdb
import numpy as np


def main(export, out):
    os.makedirs(out, exist_ok=True)
    ns = chr(92) + "N"
    c = duckdb.connect()
    R = lambda n: f"read_csv('{os.path.join(export, n + '.csv.gz')}', header=true, nullstr='{ns}', sample_size=-1, all_varchar=true)"
    c.execute(f"""CREATE TABLE q AS SELECT id::BIGINT AS question_id, cat_id::BIGINT cat_id, try_cast(fail AS BIGINT) fail,
                  try_cast(correct AS BIGINT) correct, try_cast(created_at AS TIMESTAMP) created FROM {R('questions')}""")
    c.execute(f"""CREATE TABLE deal AS SELECT question_id, count(*) n_games FROM (SELECT unnest(string_split(questions, ','))::BIGINT AS question_id
                  FROM {R('games')} WHERE questions IS NOT NULL AND questions <> '') GROUP BY 1""")
    c.execute(f"""CREATE TABLE rc AS SELECT cat_id::BIGINT cat_id, count(*) FILTER (WHERE round::INT BETWEEN 1 AND 4) r14,
                  count(*) FILTER (WHERE round::INT = 5) r5 FROM {R('matches')} GROUP BY 1""")
    d = c.sql("""SELECT q.question_id, q.cat_id, created >= TIMESTAMP '2025-01-01' AS new_item, (fail + correct) / (2.0 * n_games) AS ratio,
                        rc.r14 / nullif(rc.r5, 0) AS R_c, n_games
                 FROM q JOIN deal USING (question_id) JOIN rc USING (cat_id) WHERE n_games >= 20 AND rc.r5 > 0""").df()
    coh = d.groupby("new_item").agg(items=("ratio", "size"), median_ratio=("ratio", "median"), mean_ratio=("ratio", "mean"))
    coh.to_csv(os.path.join(out, "counter_ratio_by_cohort.csv"), float_format="%.3f")
    n = d[d.new_item]
    cat = n.groupby("cat_id").agg(median_ratio=("ratio", "median"), R_c=("R_c", "first"), items=("ratio", "size"))
    cat["expected_if_shared_pool"] = 1 + cat.R_c
    cat.sort_values("R_c").to_csv(os.path.join(out, "counter_ratio_by_category.csv"), float_format="%.3f")
    w = cat["items"].values
    X = np.column_stack([np.ones(len(cat)), cat.R_c.values])
    b = np.linalg.lstsq(X * np.sqrt(w)[:, None], cat.median_ratio.values * np.sqrt(w), rcond=None)[0]
    # learner-free bootstrap over categories for the slope
    rng = np.random.default_rng(42)
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(cat), len(cat))
        bs.append(np.linalg.lstsq(X[k] * np.sqrt(w[k])[:, None], cat.median_ratio.values[k] * np.sqrt(w[k]), rcond=None)[0][1])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    with open(os.path.join(out, "round_pools_summary.txt"), "w", encoding="utf-8") as f:
        f.write(f"items created 2025+: {len(n)} in {len(cat)} categories\n")
        f.write(f"median ratio (counter / round-5 answers): {n.ratio.median():.2f}; expected under a shared pool: "
                f"{(1 + n.R_c).median():.2f}\n")
        f.write(f"category-level weighted fit: ratio = {b[0]:.2f} + {b[1]:.3f} * R_c (95% CI of slope {lo:.3f} to {hi:.3f}; "
                f"shared pool predicts slope 1)\n")
    print(open(os.path.join(out, "round_pools_summary.txt"), encoding="utf-8").read())
    print(coh.to_string(float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
