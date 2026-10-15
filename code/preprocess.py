"""
Platform export -> analysis-ready datasets (complete records only).

    pip install duckdb pyarrow
    python preprocess.py <export_dir> <out_dir>

<export_dir> = unpacked folder mysql_2025-01-01_2026-10-05 (contains manifest.json).

Principle: a record enters a modelling table only if every field the model needs is observed.
Incomplete-but-critical information is repaired by deterministic, documented rules; nothing is
imputed with invented values.

Outputs (Parquet unless stated)
  accounts_screened.csv    accounts removed as non-human (bots, invalid id, collusion) or flagged high_volume
  kc_responses             15 binary responses per complete player-game (5 rounds x 3 questions);
                           knowledge component (KC) = category
  item_responses           round-5 responses with question_id, exposure index, previous outcome, gap
  player_games_complete    one row per player-game in which the player answered all 15 questions
  games_complete           complete human-vs-human games with consistent outcome (Elo / outcome models)
  activity_days            player x day activity from ALL games of human players (retention / churn targets)
  items                    question bank rows that occur in item_responses (Kazakh text, category, is_ai)
  preprocessing_funnel.csv row counts after each step (report in the Methods section)

Preprocessing rules
  P1  Time zone. Source timestamps are MSK (UTC+3); converted to Kazakhstan time (UTC+5).
  P2  Account screening. Non-human accounts are never modelled as learners:
        a) declared bots (table `bots`);
        b) invalid id 0;
        c) undeclared-bot signature: >= 1000 games, >= 50 games per active day, no device_type;
        d) collusion: >= 1000 games, win rate >= 0.99 or <= 0.01, distinct opponents < 1% of games.
      Human high-volume players (same volume, real device) are retained and flagged `high_volume`.
      Human responses in games against a bot are genuine and kept (flag `opp_is_bot`).
  P3  Completeness. Response tables: status 'end', exactly 5 round records, and the modelled player
      answered all 15 questions. Outcome table: additionally both players human and both complete.
      Unfinished games remain in activity_days (complete as activity events).
  P3c Unobserved practice. Round records of games against bots were not stored before April 2026,
      so part of some players' practice is unobserved. Every row carries prior_games_all (all games in
      the export) and prior_games_observed, so models account for the gap explicitly.
  P4  Outcome consistency. A recorded winner with fewer correct answers than the loser -> game removed
      from games_complete. Equal totals with a winner = time tie-break (kept, flagged).
  P5  Item identification. games.questions holds only the 3 questions of round 5 (category agreement
      with round 5 = 99.7%, rounds 1-4 = 0%). Item-level data = round 5 only; rounds 1-4 enter at KC level.
      Question ids absent from the bank or with a contradicting category are removed.
  P6  Left truncation. Players registered before 2025-01-01 have unobserved earlier history.
      `history_complete` = 1 for players registered inside the window. Primary analyses use these
      players; the others serve sensitivity analysis and model pre-training.
"""
import os
import sys

import duckdb

VOLUME_MIN_GAMES = 1000
VOLUME_GAMES_PER_DAY = 50
COLLUSION_WR = 0.99
COLLUSION_OPP_SHARE = 0.01
WINDOW_START = "2025-01-01"


def main(src: str, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='5GB'")
    funnel = []

    def step(label: str, sql: str) -> None:
        n = con.execute(sql).fetchone()[0]
        funnel.append((label, n))
        print(f"{label:<66} {n:>12,}", flush=True)

    def load(name: str, shift_cols=()) -> None:
        p = os.path.join(src, f"{name}.csv.gz")
        con.execute(f"CREATE TABLE raw_{name} AS SELECT * FROM read_csv('{p}', header=true, nullstr='\\N', sample_size=-1)")
        if shift_cols:  # P1
            con.execute(f"UPDATE raw_{name} SET " + ", ".join(f"{c} = {c} + INTERVAL 2 HOUR" for c in shift_cols))

    load("games", ["created_at", "updated_at"])
    load("matches", ["created_at"])
    load("users", ["created_at"])
    load("questions")
    load("bots")
    step("games in export", "SELECT count(*) FROM raw_games")

    # ---------------- P2 account screening ----------------
    con.execute("""
        CREATE TABLE pv AS
        SELECT id AS game_id, user_id AS uid, opponent_id AS opp_id, winner_id, status, created_at, TRUE AS is_user_side
        FROM raw_games WHERE user_id IS NOT NULL
        UNION ALL
        SELECT id, opponent_id, user_id, winner_id, status, created_at, FALSE
        FROM raw_games WHERE opponent_id IS NOT NULL""")
    con.execute("""
        CREATE TABLE acc AS
        SELECT uid, count(*) AS n_games, count(DISTINCT CAST(created_at AS DATE)) AS active_days,
               avg((winner_id = uid)::INT) AS win_rate, count(DISTINCT opp_id) AS n_opponents
        FROM pv GROUP BY uid""")
    con.execute(f"""
        CREATE TABLE screened AS
        SELECT a.uid, a.n_games, a.active_days, round(a.n_games / a.active_days, 1) AS games_per_day,
               round(a.win_rate, 3) AS win_rate, a.n_opponents, u.device_type,
               CASE
                 WHEN a.uid IN (SELECT user_id FROM raw_bots) THEN 'P2a declared bot'
                 WHEN a.uid = 0 THEN 'P2b invalid id'
                 WHEN a.n_games >= {VOLUME_MIN_GAMES}
                      AND (a.win_rate >= {COLLUSION_WR} OR a.win_rate <= 1 - {COLLUSION_WR})
                      AND a.n_opponents < {COLLUSION_OPP_SHARE} * a.n_games THEN 'P2d collusion'
                 WHEN a.n_games >= {VOLUME_MIN_GAMES} AND a.n_games / a.active_days >= {VOLUME_GAMES_PER_DAY}
                      AND u.device_type IS NULL THEN 'P2c undeclared bot signature'
                 WHEN a.n_games >= {VOLUME_MIN_GAMES} AND a.n_games / a.active_days >= {VOLUME_GAMES_PER_DAY}
                      THEN 'flag high_volume (retained)'
               END AS reason
        FROM acc a LEFT JOIN raw_users u ON u.id = a.uid""")
    con.execute("DELETE FROM screened WHERE reason IS NULL")
    con.execute(f"COPY (SELECT * FROM screened ORDER BY n_games DESC) TO '{out}/accounts_screened.csv' (HEADER)")
    con.execute("CREATE TABLE excluded AS SELECT uid FROM screened WHERE reason NOT LIKE 'flag%'")
    con.execute("CREATE TABLE high_volume AS SELECT uid FROM screened WHERE reason LIKE 'flag%'")

    con.execute("DELETE FROM pv WHERE uid IN (SELECT uid FROM excluded) OR uid = opp_id")
    step("P2 player-games of human players (all games)", "SELECT count(*) FROM pv")

    con.execute("""
        CREATE TABLE activity_days AS
        SELECT uid, CAST(created_at AS DATE) AS d, count(*) AS games FROM pv GROUP BY ALL""")

    # ---------------- P3 completeness ----------------
    con.execute("""
        CREATE TABLE mstat AS
        SELECT game_id, count(*) AS n_rows, count(DISTINCT round) AS n_rounds, max(round) AS max_round,
               sum((user_first_question IS NOT NULL AND user_second_question IS NOT NULL
                    AND user_thirty_question IS NOT NULL)::INT) AS u_full,
               sum((opponent_first_question IS NOT NULL AND opponent_second_question IS NOT NULL
                    AND opponent_thirty_question IS NOT NULL)::INT) AS o_full,
               sum(user_correct_count) AS u_total, sum(opponent_correct_count) AS o_total
        FROM raw_matches GROUP BY game_id""")
    con.execute("""
        CREATE TABLE pv2 AS
        WITH b AS (
          SELECT p.*, m.game_id IS NOT NULL AS observed,
                 coalesce(p.status = 'end' AND m.n_rows = 5 AND m.n_rounds = 5 AND m.max_round = 5, FALSE) AS shape_ok,
                 CASE WHEN p.is_user_side THEN m.u_full ELSE m.o_full END AS s_full,
                 CASE WHEN p.is_user_side THEN m.o_full ELSE m.u_full END AS o_full,
                 CASE WHEN p.is_user_side THEN m.u_total ELSE m.o_total END AS correct_total,
                 CASE WHEN p.is_user_side THEN m.o_total ELSE m.u_total END AS opp_correct_total
          FROM pv p LEFT JOIN mstat m ON m.game_id = p.game_id)
        SELECT *,
               shape_ok AND s_full = 5 AS subject_complete,
               shape_ok AND s_full = 5 AND o_full = 5 AS both_complete,
               opp_id IN (SELECT uid FROM excluded) AS opp_is_bot,
               uid IN (SELECT uid FROM high_volume) AS high_volume,
               row_number() OVER w - 1 AS prior_games_all,
               coalesce(sum(observed::INT) OVER (w ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING), 0)
                   AS prior_games_observed,
               date_diff('second', lag(created_at) OVER w, created_at) AS sec_since_prev_game
        FROM b
        WINDOW w AS (PARTITION BY uid ORDER BY created_at, game_id)""")
    step("P3a player-games with round records", "SELECT count(*) FROM pv2 WHERE observed")
    step("P3b player-games: player answered all 15 questions", "SELECT count(*) FROM pv2 WHERE subject_complete")

    # ---------------- P4 outcome table ----------------
    con.execute("""
        CREATE TABLE games_complete AS
        SELECT g.id AS game_id, g.user_id, g.opponent_id, g.winner_id, g.type AS game_type, g.created_at,
               m.u_total, m.o_total,
               CASE WHEN g.winner_id IS NULL THEN 0.5 WHEN g.winner_id = g.user_id THEN 1.0 ELSE 0.0 END AS user_score,
               (m.u_total = m.o_total AND g.winner_id IS NOT NULL) AS tie_break
        FROM raw_games g JOIN mstat m ON m.game_id = g.id
        WHERE g.status = 'end' AND m.n_rows = 5 AND m.n_rounds = 5 AND m.max_round = 5
          AND m.u_full = 5 AND m.o_full = 5
          AND g.user_id NOT IN (SELECT uid FROM excluded) AND g.opponent_id NOT IN (SELECT uid FROM excluded)
          AND g.user_id <> g.opponent_id
          AND ((g.winner_id IS NULL AND m.u_total = m.o_total)
            OR (g.winner_id = g.user_id AND m.u_total >= m.o_total)
            OR (g.winner_id = g.opponent_id AND m.o_total >= m.u_total))""")
    step("P4 human-vs-human complete games, consistent outcome", "SELECT count(*) FROM games_complete")

    # ---------------- responses ----------------
    con.execute("""
        CREATE TABLE kc0 AS
        SELECT p.game_id, p.created_at AS game_time, m.round, m.cat_id, p.uid, p.opp_id, p.opp_is_bot,
               p.high_volume, p.prior_games_all, p.prior_games_observed,
               unnest([1, 2, 3]) AS pos,
               unnest(CASE WHEN p.is_user_side
                           THEN [m.user_first_question, m.user_second_question, m.user_thirty_question]
                           ELSE [m.opponent_first_question, m.opponent_second_question, m.opponent_thirty_question]
                      END) AS y
        FROM pv2 p JOIN raw_matches m ON m.game_id = p.game_id
        WHERE p.subject_complete""")
    con.execute(f"""
        CREATE TABLE kc_responses AS
        SELECT k.*,
               (u.created_at >= TIMESTAMP '{WINDOW_START}')::INT AS history_complete,
               row_number() OVER (PARTITION BY k.uid, k.cat_id ORDER BY k.game_time, k.game_id, k.round, k.pos) AS kc_attempt,
               row_number() OVER (PARTITION BY k.uid ORDER BY k.game_time, k.game_id, k.round, k.pos) AS seq_idx
        FROM kc0 k LEFT JOIN raw_users u ON u.id = k.uid""")
    step("KC-level responses", "SELECT count(*) FROM kc_responses")

    # ---------------- P5 item level ----------------
    con.execute("""
        CREATE TABLE it0 AS
        SELECT k.*, string_split(g.questions, ',')[k.pos]::BIGINT AS question_id
        FROM kc_responses k JOIN raw_games g ON g.id = k.game_id
        WHERE k.round = 5 AND g.questions IS NOT NULL AND len(string_split(g.questions, ',')) = 3""")
    step("P5a round-5 responses with stored question ids", "SELECT count(*) FROM it0")
    con.execute("""
        CREATE TABLE it1 AS SELECT i.* FROM it0 i JOIN raw_questions q ON q.id = i.question_id
        WHERE q.cat_id = i.cat_id OR q.cat_id = 0""")
    step("P5b id present in bank, category consistent", "SELECT count(*) FROM it1")
    con.execute("""
        CREATE TABLE item_responses AS
        SELECT *,
               row_number() OVER w AS exposure,
               lag(y) OVER w AS prev_y,
               date_diff('second', lag(game_time) OVER w, game_time) AS gap_seconds
        FROM it1 WINDOW w AS (PARTITION BY uid, question_id ORDER BY game_time, game_id)""")
    con.execute("""
        CREATE TABLE items AS
        SELECT q.id AS question_id, q.cat_id, q.questions AS text_kk, q.is_ai,
               count(r.y) AS n_responses, avg(r.y) AS p_correct
        FROM raw_questions q JOIN item_responses r ON r.question_id = q.id GROUP BY ALL""")

    # ---------------- player-game view ----------------
    con.execute(f"""
        CREATE TABLE player_games_complete AS
        SELECT p.game_id, p.created_at, p.uid, p.opp_id, p.opp_is_bot, p.high_volume,
               CASE WHEN p.winner_id IS NULL THEN 0.5 WHEN p.winner_id = p.uid THEN 1.0 ELSE 0.0 END AS score,
               p.correct_total, p.opp_correct_total, p.both_complete,
               p.prior_games_all, p.prior_games_observed, p.sec_since_prev_game,
               (u.created_at >= TIMESTAMP '{WINDOW_START}')::INT AS history_complete
        FROM pv2 p LEFT JOIN raw_users u ON u.id = p.uid
        WHERE p.subject_complete""")

    for name in ("kc_responses", "item_responses", "player_games_complete", "games_complete", "activity_days", "items"):
        con.execute(f"COPY {name} TO '{out}/{name}.parquet' (FORMAT parquet)")

    step("item-level responses (final)", "SELECT count(*) FROM item_responses")
    step("distinct items", "SELECT count(*) FROM items")
    step("items with >= 30 responses", "SELECT count(*) FROM items WHERE n_responses >= 30")
    step("human players with >= 1 complete player-game", "SELECT count(DISTINCT uid) FROM player_games_complete")
    step("  of them registered 2025+ (history_complete)",
         "SELECT count(DISTINCT uid) FROM player_games_complete WHERE history_complete = 1")
    step("KC responses, history_complete", "SELECT count(*) FROM kc_responses WHERE history_complete = 1")
    step("item responses, history_complete", "SELECT count(*) FROM item_responses WHERE history_complete = 1")
    step("accounts excluded as non-human", "SELECT count(*) FROM excluded")
    step("human accounts flagged high_volume", "SELECT count(*) FROM high_volume")

    con.execute("CREATE TABLE funnel(step VARCHAR, n BIGINT)")
    con.executemany("INSERT INTO funnel VALUES (?, ?)", funnel)
    con.execute(f"COPY funnel TO '{out}/preprocessing_funnel.csv' (HEADER)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
