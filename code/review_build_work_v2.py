"""
Assemble the inputs of the RQ2/RQ3 analysis under the revised primary specification (second review):
  - leak-free online Elo (ratings updated after each game) for M4, M5, M5a  -> review_rq2_leakfree.py
  - one counterfactual protocol for all models: direct item-memory (the item's identity as an earlier encounter is
    removed, the amount, timing and outcomes of practice are kept):
        GBM: memory features reset (as before); PFA and DAS3H without item windows: no item-specific feature -> 0;
        DAS3H + item windows: item windows zeroed, category windows kept  -> review_das3h_direct.py;
        DKT, SAKT, DKT-time: earlier tokens of the item masked by the category placeholder -> review_deep_direct_rows.py
The resulting <work_v2>/rq2 tree has the same layout as <work>/rq2, so all pipeline scripts run on it unchanged.

    python review_build_work_v2.py <work> <work_v2>
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rq2_models import LAG_BINS, lag_bin  # noqa: E402

KEY = ["uid", "game_id", "question_id"]


def implied_table(df, preds, cfs, lag, prev, seen):
    rows = []
    for k in preds:
        imp = (df[k] - df[cfs[k]]).values if k in cfs else np.zeros(len(df))
        for lab, m_ in [("all", seen)] + [(f"lag {b}", seen & (lag == b)) for _, _, b in LAG_BINS] + \
                       [(f"after correct, lag {b}", seen & (lag == b) & (prev == 1)) for _, _, b in LAG_BINS] + \
                       [(f"after error, lag {b}", seen & (lag == b) & (prev == 0)) for _, _, b in LAG_BINS]:
            rows.append(dict(model=k, subset=lab, n=int(m_.sum()), implied_effect=float(imp[m_].mean()) if m_.any() else np.nan,
                             item_specific_path=k in cfs))
    return pd.DataFrame(rows)


def build(work, v2, tag):
    sub = (lambda r: r) if tag == "" else (lambda r: os.path.join(r, "fold_" + tag))
    src, out = sub(os.path.join(work, "rq2")), sub(os.path.join(v2, "rq2"))
    leak, d3 = sub(os.path.join(work, "review", "rq2_leakfree")), sub(os.path.join(work, "review", "cf_das3h_direct"))
    deep = os.path.join(v2, "stage_deep", "F3" if tag == "" else "fold_" + tag)
    os.makedirs(out, exist_ok=True)
    # stage 1: leak-free predictions; PFA under the direct protocol implies 0
    t = pd.read_parquet(os.path.join(leak, "rq2_test_predictions.parquet"))
    t["M3 PFA | cf unseen"] = t["M3 PFA"]
    t.to_parquet(os.path.join(out, "rq2_test_predictions.parquet"), index=False)
    shutil.copy(os.path.join(leak, "rq2_metrics_test.csv"), out)
    shutil.copy(os.path.join(leak, "rq2_lagshift_M5.csv"), out)
    shutil.copy(os.path.join(leak, "rq2_gbm_lag_shift.csv"), out)
    preds = [c for c in t.columns if c.startswith("M") and "| cf" not in c]
    cfs = {k: k + " | cf unseen" for k in preds if k + " | cf unseen" in t.columns}
    seen = t.exposure.values > 1
    lag = lag_bin(np.where(seen, np.expm1(t.log_gap_h.values), -1))
    implied_table(t, preds, cfs, lag, t.prev_y.values, seen).to_csv(os.path.join(out, "rq2_implied_effects.csv"), index=False, float_format="%.5f")
    # DAS3H: M6 from the main run with implied 0 (no item-specific feature); M6b (fit, predictions and implied) from the direct run
    a = pd.read_parquet(os.path.join(src, "rq2_rows_das3h.parquet"))
    b = pd.read_parquet(os.path.join(d3, "rq2_rows_das3h.parquet"))
    assert (a[KEY].values == b[KEY].values).all()
    a["implied | M6 DAS3H"] = 0.0
    for c in ("p | M6b DAS3H + item windows", "implied | M6b DAS3H + item windows"):
        a[c] = b[c].values
    a.to_parquet(os.path.join(out, "rq2_rows_das3h.parquet"), index=False)
    m_old = pd.read_csv(os.path.join(src, "rq2_metrics_test_das3h.csv"))
    m_new = pd.read_csv(os.path.join(d3, "rq2_metrics_test_das3h.csv"))
    pd.concat([m_old[m_old.model != "M6b DAS3H + item windows"], m_new]).to_csv(os.path.join(out, "rq2_metrics_test_das3h.csv"), index=False)
    shutil.copy(os.path.join(d3, "rq2_lagshift_M6b.csv"), out)
    sd = a.exposure.values > 1
    tab = []
    for name in ("M6 DAS3H", "M6b DAS3H + item windows"):
        imp = a["implied | " + name].values; lg = a.lag_bin.values; pv = a.prev_y.values
        for lab, m_ in [("all", sd)] + [(f"lag {bb}", sd & (lg == bb)) for _, _, bb in LAG_BINS] + \
                       [(f"after correct, lag {bb}", sd & (lg == bb) & (pv == 1)) for _, _, bb in LAG_BINS] + \
                       [(f"after error, lag {bb}", sd & (lg == bb) & (pv == 0)) for _, _, bb in LAG_BINS]:
            tab.append(dict(model=name, subset=lab, n=int(m_.sum()), implied_effect=float(imp[m_].mean()) if m_.any() else np.nan,
                            item_specific_path=True))
    pd.DataFrame(tab).to_csv(os.path.join(out, "rq2_implied_effects_das3h.csv"), index=False, float_format="%.5f")
    # deep: predictions unchanged; implied rows under the direct protocol
    for f in os.listdir(src):
        if (f.startswith("rq2_rows_deep_") and f.count("_") == 3) or f in ("rq2_metrics_test_deep.csv", "rq2_lagshift_DKT-time.csv"):
            shutil.copy(os.path.join(src, f), out)
    for f in os.listdir(deep):
        shutil.copy(os.path.join(deep, f), out)
    print(tag or "F3", "built:", sorted(os.listdir(out)))


def main(work, v2):
    for d in ("rq1", "design_checks"):
        dst = os.path.join(v2, d)
        if not os.path.exists(dst):
            shutil.copytree(os.path.join(work, d), dst)
    for tag in ("", "2025-07", "2025-10"):
        build(work, v2, tag)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
