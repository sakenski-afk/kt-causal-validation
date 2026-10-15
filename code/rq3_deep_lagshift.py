"""
RQ3 helper: model-implied forgetting curve of DKT-time (saved weights), by previous outcome.

    python rq3_deep_lagshift.py <rq2_dir> [--window 200]

Uses deep_tokens.npz and deep_DKT-time[_L500].pt from rq2_deep.py. For the same seeded sample of
test re-exposures as in rq2_deep.py, the context excludes the learner's earlier tokens with the
item (so the item path runs through the target time features only), and the target item-memory
features are set to 'one earlier encounter h hours ago' versus 'never seen'. Implied effect(h) =
mean difference of predicted probabilities. DKT and SAKT have no item path (RQ2) and are not queried.
Output: rq2_lagshift_DKT-time[_L500].csv (model, prev_y, hours, n, implied).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq2_deep as R  # noqa: E402
from rq2_models import GRID_H  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rq2"); ap.add_argument("--window", type=int, default=200)
    a = ap.parse_args()
    R.L, R.STRIDE, R.BATCH = a.window, a.window // 2, 64
    sfx = "" if a.window == 200 else f"_L{a.window}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {k: (v.item() if v.ndim == 0 else v) for k, v in np.load(os.path.join(a.rq2, "deep_tokens.npz"), allow_pickle=True).items()}
    model = R.DKT(data["n_item_tok"], data["n_cat"], time=True).to(dev)
    model.load_state_dict(torch.load(os.path.join(a.rq2, f"deep_DKT-time{sfx}.pt"), map_location=dev))
    model.eval()
    te5 = data["is5"] & (data["split"] == 2)
    seen_te = np.flatnonzero(te5 & (data["exposure"] > 1))
    rng = np.random.default_rng(R.SEED)
    sample_idx = np.sort(rng.choice(seen_te, size=min(R.N_IMPLIED, len(seen_te)), replace=False))
    starts = np.repeat(data["starts"], data["ends"] - data["starts"])
    lists = []
    for j in sample_idx:
        s = starts[j]
        full = np.arange(max(s, j - 5000), j + 1)
        keep = ~((data["item"][full] == data["item"][j]) & (full < j))
        lists.append(full[keep][-R.L:])
    prev = data["prev_y"][sample_idx]

    def predict(target_feats):
        out = np.empty(len(lists), np.float32)
        with torch.no_grad():
            for b0 in range(0, len(lists), 256):
                lb = lists[b0:b0 + 256]
                item, cat, y, feats = R.batchify(data, [(0, len(x)) for x in lb], idx_lists=lb)
                for r, x in enumerate(lb):
                    feats[r, len(x) - 1, 1:] = target_feats
                it, ct, yy, ff = R.to_dev(item, cat, y, feats, dev=dev)
                p = torch.sigmoid(model(it, ct, yy, ff, ff[..., 1:])).cpu().numpy()
                for r, x in enumerate(lb):
                    out[b0 + r] = p[r, len(x) - 1]
        return out

    p0 = predict(np.zeros(3, np.float32))
    rows = []
    for h in GRID_H:
        ph = predict(np.array([np.log1p(h), np.log1p(1.0), 1.0], np.float32))
        for pv in (1, 0):
            m = prev == pv
            rows.append(dict(model=f"M7 DKT-time{sfx}", prev_y=pv, hours=h, n=int(m.sum()), implied=float((ph - p0)[m].mean())))
        print(h, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.rq2, f"rq2_lagshift_DKT-time{sfx}.csv"), index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
