"""
Descriptive statistics and figures for the Data section.

    python eda.py <preprocessed_dir> <out_dir>

<preprocessed_dir> = output folder of preprocess.py
"""
import os
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.dpi": 300,
    "legend.frameon": False, "lines.linewidth": 2,
})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main(src, out):
    os.makedirs(out, exist_ok=True)
    c = duckdb.connect()
    P = lambda n: f"'{os.path.join(src, n + '.parquet')}'"
    rows = []

    def add(group, name, sql, fmt="{:,.0f}"):
        v = c.execute(sql).fetchone()
        val = fmt.format(*v) if isinstance(v, tuple) else fmt.format(v)
        rows.append((group, name, val))

    for coh, cond in (("All players", "TRUE"), ("Cohort 2025+ (complete history)", "history_complete = 1")):
        add(coh, "Players", f"SELECT count(DISTINCT uid) FROM {P('player_games_complete')} WHERE {cond}")
        add(coh, "Complete player-games", f"SELECT count(*) FROM {P('player_games_complete')} WHERE {cond}")
        add(coh, "KC-level responses", f"SELECT count(*) FROM {P('kc_responses')} WHERE {cond}")
        add(coh, "Item-level responses", f"SELECT count(*) FROM {P('item_responses')} WHERE {cond}")
        add(coh, "Mean accuracy, KC level", f"SELECT avg(y) FROM {P('kc_responses')} WHERE {cond}", "{:.3f}")
        add(coh, "Player-games per player, median [IQR]",
            f"SELECT quantile_cont(n,0.5), quantile_cont(n,0.25), quantile_cont(n,0.75) FROM "
            f"(SELECT uid, count(*) n FROM {P('player_games_complete')} WHERE {cond} GROUP BY 1)",
            "{:.0f} [{:.0f}–{:.0f}]")
        add(coh, "Share of re-exposures (item level)",
            f"SELECT avg((exposure > 1)::INT) FROM {P('item_responses')} WHERE {cond}", "{:.3f}")
        add(coh, "Observed share of prior practice, mean",
            f"SELECT avg(prior_games_observed / prior_games_all) FROM {P('player_games_complete')} "
            f"WHERE {cond} AND prior_games_all > 0", "{:.3f}")
    add("Items", "Distinct items", f"SELECT count(*) FROM {P('items')}")
    add("Items", "Items with >= 30 responses", f"SELECT count(*) FROM {P('items')} WHERE n_responses >= 30")
    add("Items", "Responses per item, median [IQR]",
        f"SELECT quantile_cont(n_responses,0.5), quantile_cont(n_responses,0.25), quantile_cont(n_responses,0.75) FROM {P('items')}",
        "{:.0f} [{:.0f}–{:.0f}]")
    add("Items", "Item p-correct, mean (SD), items with >= 30 responses",
        f"SELECT avg(p_correct), stddev(p_correct) FROM {P('items')} WHERE n_responses >= 30", "{:.3f} ({:.3f})")
    add("Items", "AI-generated items (is_ai = 1)", f"SELECT sum(is_ai) FROM {P('items')}")
    add("Items", "Knowledge components (categories)", f"SELECT count(DISTINCT cat_id) FROM {P('kc_responses')}")
    add("Games", "Complete human-vs-human games", f"SELECT count(*) FROM {P('games_complete')}")
    add("Games", "Draw share", f"SELECT avg((user_score = 0.5)::INT) FROM {P('games_complete')}", "{:.3f}")
    add("Games", "Observation window", f"SELECT min(created_at)::DATE::VARCHAR, max(created_at)::DATE::VARCHAR FROM {P('games_complete')}", "{} – {}")

    with open(os.path.join(out, "table1_descriptives.csv"), "w", encoding="utf-8") as f:
        f.write("group,statistic,value\n")
        for g, n, v in rows:
            f.write(f'"{g}","{n}","{v}"\n')
    for r in rows:
        print(" | ".join(r))

    # ---- Figure 1: monthly activity ----
    df = c.sql(f"""SELECT date_trunc('month', d) m, count(DISTINCT uid) mau, sum(games) g
                   FROM {P('activity_days')} WHERE d < DATE '2026-10-01' GROUP BY 1 ORDER BY 1""").df()
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.plot(df.m, df.mau, color=BLUE, marker="o", ms=4)
    ax.set_ylabel("Monthly active players")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylim(0, df.mau.max() * 1.1)
    ax.set_title("Monthly active human players, Jan 2025 – Sep 2026", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_monthly_active.png"))
    plt.close(fig)

    # ---- Figure 2: accuracy by exposure index ----
    df = c.sql(f"""SELECT least(exposure, 6) e, count(*) n, sum(y) k FROM {P('item_responses')}
                   GROUP BY 1 ORDER BY 1""").df()
    lo, hi = wilson(df.k.values, df.n.values)
    p = df.k / df.n
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.errorbar(df.e, p, yerr=[p - lo, hi - p], color=BLUE, marker="o", ms=5, capsize=3)
    ax.set_xticks(df.e)
    ax.set_xlim(0.6, 6.8)
    ax.set_xticklabels([str(int(x)) if x < 6 else "6+" for x in df.e])
    ax.set_xlabel("Exposure to the same item (k-th encounter)")
    ax.set_ylabel("P(correct)")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Accuracy by repeated exposure", loc="left", color=INK, fontsize=10)
    for x, y, n in zip(df.e, p, df.n):
        ax.annotate(f"n={n/1000:.0f}k", (x, y), textcoords="offset points", xytext=(10, -12),
                    ha="left", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_exposure.png"))
    plt.close(fig)

    # ---- Figure 3: forgetting curve ----
    edges = [0, 3600, 6 * 3600, 86400, 3 * 86400, 7 * 86400, 14 * 86400, 30 * 86400, 60 * 86400, 120 * 86400, 10**9]
    labels = ["<1h", "1–6h", "6–24h", "1–3d", "3–7d", "1–2w", "2–4w", "1–2m", "2–4m", ">4m"]
    case = "CASE " + " ".join(f"WHEN gap_seconds < {edges[i+1]} THEN {i}" for i in range(len(labels))) + " END"
    df = c.sql(f"""SELECT prev_y, {case} b, count(*) n, sum(y) k FROM {P('item_responses')}
                   WHERE exposure = 2 GROUP BY ALL ORDER BY 1, 2""").df()
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    for prev, col, name in ((1, BLUE, "Previous answer correct"), (0, ORANGE, "Previous answer incorrect")):
        d = df[df.prev_y == prev]
        lo, hi = wilson(d.k.values, d.n.values)
        pp = d.k / d.n
        ax.errorbar(d.b, pp, yerr=[pp - lo, hi - pp], color=col, marker="o", ms=4, capsize=2, label=name)
        ax.annotate(name, (d.b.iloc[-1], pp.iloc[-1]), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.4, len(labels) + 2.2)
    ax.set_xlabel("Time since previous encounter with the same item")
    ax.set_ylabel("P(correct) at 2nd encounter")
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Retention of item-specific knowledge between encounters", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_forgetting.png"))
    plt.close(fig)
    df.to_csv(os.path.join(out, "fig3_data.csv"), index=False)

    # ---- Figure 4: weekly retention, cohort 2025+ ----
    df = c.sql(f"""
        WITH f AS (SELECT a.uid, min(a.d) d0 FROM {P('activity_days')} a
                   JOIN (SELECT DISTINCT uid FROM {P('player_games_complete')} WHERE history_complete = 1) h USING (uid)
                   GROUP BY 1 HAVING min(a.d) < DATE '2026-05-01'),
             w AS (SELECT DISTINCT a.uid, (a.d - f.d0) // 7 wk FROM {P('activity_days')} a JOIN f USING (uid))
        SELECT wk, count(*) n, (SELECT count(*) FROM f) base FROM w WHERE wk BETWEEN 0 AND 20 GROUP BY 1 ORDER BY 1""").df()
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    ax.plot(df.wk, df.n / df.base, color=BLUE, marker="o", ms=3)
    ax.set_xlabel("Weeks since first game")
    ax.set_xticks(range(0, 21, 2))
    ax.set_ylabel("Share of cohort active")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Weekly activity after first game, cohort 2025+ (n={int(df.base.iloc[0]):,})", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_retention.png"))
    plt.close(fig)
    df.to_csv(os.path.join(out, "fig4_data.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
