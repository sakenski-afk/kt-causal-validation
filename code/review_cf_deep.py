"""
Reviewer check: standardized counterfactual protocols for the deep models (test window F3, saved weights).

    python review_cf_deep.py <rq2_dir> <out_dir>

For the same seeded sample of test re-exposures as rq2_deep.py, three queries are compared:
  full     full-history deletion (as in the paper): the learner's earlier round-5 tokens with this item are removed
           from the context (sequence shifts; the context is refilled with older tokens);
  direct   item-identity masking: the earlier tokens stay in place with their responses, times and categories, but their
           item id is replaced by the category placeholder used for rounds 1-4 and their item-memory features are zeroed;
           for DKT-time the target item-memory features are also set to 'never seen';
  target   (DKT-time only) only the target item-memory features are set to 'never seen'; context unchanged.
Per-sample p_act / p_cf are written; aggregation by lag bin is done in review_cf_summary.py.
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rq2_deep as Dm  # noqa: E402
from rq2_models import lag_bin  # noqa: E402


def run(model, name, data, dev, sample_idx, protocol):
    L = Dm.L
    starts = np.repeat(data["starts"], data["ends"] - data["starts"])
    nq_ph = data["n_item_tok"] - data["n_cat"]  # placeholder id = nq + 1 + cat
    ctx_lists, masks = [], []
    for j in sample_idx:
        s = starts[j]
        if protocol == "full":
            lo = max(s, j - 5000)
            full = np.arange(lo, j + 1)
            keep = ~((data["item"][full] == data["item"][j]) & (full < j))
            ctx = full[keep][-L:]
            masks.append(np.zeros(len(ctx), bool))
        else:
            ctx = np.arange(max(s, j - L + 1), j + 1)
            masks.append((data["item"][ctx] == data["item"][j]) & (ctx < j) if protocol == "direct" else np.zeros(len(ctx), bool))
        ctx_lists.append(ctx)
    res = np.empty(len(sample_idx), np.float32)
    B = 2 * Dm.BATCH
    with torch.no_grad():
        for b0 in range(0, len(ctx_lists), B):
            lb = ctx_lists[b0:b0 + B]
            item, cat, y, feats = Dm.batchify(data, [(0, len(x)) for x in lb], idx_lists=lb)
            for r, x in enumerate(lb):
                mk = masks[b0 + r]
                if mk.any():
                    item[r, :len(x)][mk] = nq_ph + cat[r, :len(x)][mk]
                    feats[r, :len(x)][mk, 1:] = 0.0
                if name == "DKT-time":
                    feats[r, len(x) - 1, 1:] = 0.0
            it, ct, yy, ff = Dm.to_dev(item, cat, y, feats, dev=dev)
            p = torch.sigmoid(model(it, ct, yy, ff, ff[..., 1:])).cpu().numpy()
            for r, x in enumerate(lb):
                res[b0 + r] = p[r, len(x) - 1]
    return res


def main(rq2, out):
    os.makedirs(out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {k: (v.item() if v.ndim == 0 else v) for k, v in np.load(os.path.join(rq2, "deep_tokens.npz"), allow_pickle=True).items()}
    te5 = data["is5"] & (data["split"] == 2)
    seen_te = np.flatnonzero(te5 & (data["exposure"] > 1))
    rng = np.random.default_rng(Dm.SEED)
    sample_idx = np.sort(rng.choice(seen_te, size=min(Dm.N_IMPLIED, len(seen_te)), replace=False))
    lb = lag_bin(np.where(data["exposure"] > 1, data["gap_h"], -1))[sample_idx]
    rows = []
    for name in ("DKT", "DKT-time", "SAKT"):
        model = (Dm.DKT(data["n_item_tok"], data["n_cat"], time=(name == "DKT-time")) if name.startswith("DKT")
                 else Dm.SAKT(data["n_item_tok"], data["n_cat"])).to(dev)
        model.load_state_dict(torch.load(os.path.join(rq2, f"deep_{name}.pt"), map_location=dev))
        model.eval()
        p_act = run(model, "ACT", data, dev, sample_idx, "actual")  # 'ACT': no target-feature reset
        for protocol in (("full", "direct", "target") if name == "DKT-time" else ("full", "direct")):
            if protocol == "target":
                p_cf = run(model, name, data, dev, sample_idx, "actual")
            else:
                p_cf = run(model, name, data, dev, sample_idx, protocol)
            imp = p_act - p_cf
            print(name, protocol, "mean implied:", round(float(imp.mean()), 5), flush=True)
            rows.append(pd.DataFrame({"model": f"M7 {name}", "protocol": protocol, "uid": data["uid"][sample_idx],
                                      "game_id": data["game_id"][sample_idx], "question_id": data["question_id"][sample_idx],
                                      "lag_bin": lb, "prev_y": data["prev_y"][sample_idx], "implied": imp}))
    pd.concat(rows, ignore_index=True).to_parquet(os.path.join(out, "cf_protocols_deep_rows.parquet"), index=False)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
