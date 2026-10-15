"""
Direct item-memory counterfactual for the deep models, written in the pipeline's format (rq2_implied_rows_deep_<name>.parquet
with uid, game_id, question_id, lag_bin, prev_y, implied, p_act, p_cf; rq2_implied_effects_deep.csv), so that the
downstream scripts can use the unified protocol unchanged.

    python review_deep_direct_rows.py <window_dir_with_weights_and_tokens> <out_dir> [--models DKT,DKT-time,SAKT]

Same seeded sample of test re-exposures and the same saved weights as rq2_deep.py; protocol as in
review_cf_deep.py ('direct': earlier tokens of the item kept in place, item identity replaced by the category
placeholder, their item-memory features zeroed; DKT-time: target item-memory features set to 'never seen').
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq2_deep as Dm  # noqa: E402
from review_cf_deep import run  # noqa: E402
from rq2_models import LAG_BINS, lag_bin  # noqa: E402


def main(src, out, models):
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {k: (v.item() if v.ndim == 0 else v) for k, v in np.load(os.path.join(src, "deep_tokens.npz"), allow_pickle=True).items()}
    te5 = data["is5"] & (data["split"] == 2)
    seen_te = np.flatnonzero(te5 & (data["exposure"] > 1))
    rng = np.random.default_rng(Dm.SEED)
    sample_idx = np.sort(rng.choice(seen_te, size=min(Dm.N_IMPLIED, len(seen_te)), replace=False))
    lb = lag_bin(np.where(data["exposure"] > 1, data["gap_h"], -1))[sample_idx]
    prev = data["prev_y"][sample_idx]
    imp_rows = []
    for name in models:
        model = (Dm.DKT(data["n_item_tok"], data["n_cat"], time=(name == "DKT-time")) if name.startswith("DKT")
                 else Dm.SAKT(data["n_item_tok"], data["n_cat"])).to(dev)
        model.load_state_dict(torch.load(os.path.join(src, f"deep_{name}.pt"), map_location=dev))
        model.eval()
        p_act = run(model, "ACT", data, dev, sample_idx, "actual")
        p_cf = run(model, name, data, dev, sample_idx, "direct")
        imp = p_act - p_cf
        pd.DataFrame({"uid": data["uid"][sample_idx], "game_id": data["game_id"][sample_idx], "question_id": data["question_id"][sample_idx],
                      "lag_bin": lb, "prev_y": prev, "implied": imp, "p_act": p_act, "p_cf": p_cf}
                     ).to_parquet(os.path.join(out, f"rq2_implied_rows_deep_{name}.parquet"), index=False)
        for lab, msk in [("all", np.ones(len(sample_idx), bool))] + [(f"lag {b}", lb == b) for _, _, b in LAG_BINS] + \
                        [(f"after correct, lag {b}", (lb == b) & (prev == 1)) for _, _, b in LAG_BINS] + \
                        [(f"after error, lag {b}", (lb == b) & (prev == 0)) for _, _, b in LAG_BINS]:
            imp_rows.append(dict(model=f"M7 {name}", subset=lab, n=int(msk.sum()),
                                 implied_effect=float(imp[msk].mean()) if msk.any() else np.nan, item_specific_path=True))
        print(src, name, "direct implied:", round(float(imp.mean()), 5), flush=True)
    pd.DataFrame(imp_rows).to_csv(os.path.join(out, "rq2_implied_effects_deep.csv"), index=False, float_format="%.5f")


if __name__ == "__main__":
    a = sys.argv
    models = a[a.index("--models") + 1].split(",") if "--models" in a else ["DKT", "DKT-time", "SAKT"]
    main(a[1], a[2], models)
