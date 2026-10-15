"""
RQ2 stage 2b: deep knowledge-tracing models and their implied re-exposure effects.

    python rq2_deep_data.py <clean_dir> <out_dir>      (first: token cache, separate process)
    python rq2_deep.py <clean_dir> <out_dir> [--models DKT,DKT-time,SAKT] [--epochs 12] [--window 200] [--batch 256]
    --window: context length L (stride L/2); 500 = sensitivity analysis. Weights, per-row test
    predictions and per-sample implied effects are saved (suffix _L<window> when L != 200).

Token stream per learner = all KC-level responses before 2026-04-01 (rounds 1-5), ordered by
(game_time, game_id, round, pos). Round-5 tokens carry the item id; rounds 1-4 only the category
(item token = category placeholder). Predictions are scored on round-5 tokens only, as for the other
models. Split as in rq2_models.py: train labels < 2025-10-01, validation labels < 2026-01-01
(early stopping on val AUC), test labels 2026-01-01 .. 2026-03-31.

Models (d = 128, seed 42, Adam 1e-3, batch 256 windows, windows of 200 tokens):
  DKT       LSTM over interaction embeddings; prediction for token j from h_{j-1} and the target item embedding.
  DKT-time  DKT + time features (Nagatani et al. 2019 style): input - log1p hours since previous token,
            log1p hours since this learner's previous encounter with this item, log1p previous encounters;
            target side - the same item-memory features for the target item.
  SAKT      one self-attention block (8 heads), queries = target item, keys/values = past interactions.
Evaluation windows overlap (stride 100); each token is scored with at least 100 tokens of context.

Model-implied re-exposure effect (same query as for all other models): for a seeded sample of test
re-exposures, p(last 200 tokens ending at the target) - p(the same after removing the learner's earlier
round-5 tokens with this item; for DKT-time the target item-memory features set to 'never seen').
Encounters outside the 200-token context are invisible to DKT/SAKT by construction - part of the result.
Outputs follow the stage-1 format: rq2_metrics_test_deep.csv, rq2_implied_effects_deep.csv.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402
from rq2_models import LAG_BINS, lag_bin  # noqa: E402

END, T_VAL, T_TEST = "2026-04-01", "2025-10-01", "2026-01-01"
L, STRIDE, D, SEED = 200, 100, 128, 42
BATCH = 256
LR, DROPOUT, REWEIGHT = 1e-3, 0.2, 1.0  # tuning knobs (see --lr, --dropout, --reweight)
N_IMPLIED = 40000


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ----------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------
class DKT(nn.Module):
    def __init__(self, n_item, n_cat, time=False):
        super().__init__()
        self.time = time
        self.e_item, self.e_cat, self.e_resp = nn.Embedding(n_item, D, padding_idx=0), nn.Embedding(n_cat, D), nn.Embedding(2, D)
        nf = 4 if time else 0
        self.inp = nn.Linear(D + nf, D)
        self.lstm = nn.LSTM(D, D, batch_first=True)
        self.out = nn.Sequential(nn.Linear(2 * D + (3 if time else 0), D), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(D, 1))

    def forward(self, item, cat, y, feats, tfeats):
        q = self.e_item(item) + self.e_cat(cat)
        x = q + self.e_resp(y)
        if self.time:
            x = torch.cat([x, feats], -1)
        h, _ = self.lstm(self.inp(x))
        h = torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], 1)  # state BEFORE token j
        z = [h, q] + ([tfeats] if self.time else [])
        return self.out(torch.cat(z, -1)).squeeze(-1)


class SAKT(nn.Module):
    def __init__(self, n_item, n_cat):
        super().__init__()
        self.e_item, self.e_cat, self.e_resp = nn.Embedding(n_item, D, padding_idx=0), nn.Embedding(n_cat, D), nn.Embedding(2, D)
        self.pos = nn.Embedding(L + 1, D)
        self.att = nn.MultiheadAttention(D, 8, dropout=DROPOUT, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(D), nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(D, D))
        self.out = nn.Linear(D, 1)

    def forward(self, item, cat, y, feats, tfeats):
        B_, T = item.shape
        q = self.e_item(item) + self.e_cat(cat)
        kv = q + self.e_resp(y)
        # interaction j-1 is visible when predicting j: shift keys right by one, slot 0 = empty
        kv = torch.cat([torch.zeros_like(kv[:, :1]), kv[:, :-1]], 1) + self.pos(torch.arange(T, device=item.device))[None]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=item.device), 1)
        a, _ = self.att(q, kv, kv, attn_mask=mask)
        x = self.ln1(a + q)
        x = self.ln2(x + self.ffn(x))
        return self.out(x).squeeze(-1)


# ----------------------------------------------------------------------------------------------
def windows(starts, ends, stride, upto_mask=None):
    """(start, end) token ranges of length <= L for every learner sequence."""
    out = []
    for s, e in zip(starts, ends):
        if upto_mask is not None:
            e = s + int(upto_mask[s:e].sum())
        if e <= s:
            continue
        a = s
        while True:
            out.append((a, min(a + L, e)))
            if a + L >= e:
                break
            a += stride
    return out


def batchify(data, wins, idx_lists=None):
    T = max(b - a for a, b in wins)
    n = len(wins)
    item = np.zeros((n, T), np.int64); cat = np.zeros((n, T), np.int64); y = np.zeros((n, T), np.int64)
    feats = np.zeros((n, T, 4), np.float32)
    for r, (a, b) in enumerate(wins):
        ix = np.arange(a, b) if idx_lists is None else idx_lists[r]
        k = len(ix)
        item[r, :k], cat[r, :k], y[r, :k], feats[r, :k] = data["item"][ix], data["cat"][ix], data["y"][ix], data["feats"][ix]
    return item, cat, y, feats


def to_dev(*arrs, dev):
    return [torch.as_tensor(a, device=dev) for a in arrs]


def predict_windows(model, data, wins, dev, keep_from):
    """Predictions for tokens of each window; keep positions >= keep_from[r] within window r."""
    model.eval()
    preds = np.full(len(data["y"]), np.nan, np.float32)
    with torch.no_grad():
        for b0 in range(0, len(wins), 2 * BATCH):
            wb = wins[b0:b0 + 2 * BATCH]
            item, cat, y, feats = batchify(data, wb)
            it, ct, yy, ff = to_dev(item, cat, y, feats, dev=dev)
            p = torch.sigmoid(model(it, ct, yy, ff, ff[..., 1:])).cpu().numpy()
            for r, (a, b) in enumerate(wb):
                k0 = keep_from[b0 + r]
                preds[a + k0:b] = p[r, k0:b - a]
    return preds


def eval_auc(model, data, dev, upto_split):
    mask = data["split"] <= upto_split
    wins = windows(data["starts"], data["ends"], STRIDE, upto_mask=mask)
    seq_start = {s for s in data["starts"]}
    keep = [0 if a in seq_start else STRIDE for a, _ in wins]
    p = predict_windows(model, data, wins, dev, keep)
    sel = data["is5"] & (data["split"] == upto_split) & ~np.isnan(p)
    return roc_auc_score(data["y"][sel], p[sel]), p


def train_model(name, data, dev, epochs):
    set_seed(SEED)
    n_item, n_cat = data["n_item_tok"], data["n_cat"]
    model = (DKT(n_item, n_cat, time=(name == "DKT-time")) if name.startswith("DKT") else SAKT(n_item, n_cat)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    train_wins = windows(data["starts"], data["ends"], L, upto_mask=data["split"] == 0)
    best, best_state, bad = -1, None, 0
    rng = np.random.default_rng(SEED)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_wins))
        tot = 0.0
        for b0 in range(0, len(order), BATCH):
            wb = [train_wins[k] for k in order[b0:b0 + BATCH]]
            item, cat, y, feats = batchify(data, wb)
            lab_mask = np.zeros(item.shape, bool)
            for r, (a, b) in enumerate(wb):
                lab_mask[r, :b - a] = data["is5"][a:b]
            lab_mask[:, 0] = False  # no context for the first token of a window
            wt = np.ones(item.shape, np.float32)
            if REWEIGHT != 1.0:  # up-weight round-5 targets whose item already occurred earlier in the window
                for r, (a, b) in enumerate(wb):
                    seen_items = set()
                    it_w, is5_w = data["item"][a:b], data["is5"][a:b]
                    for k in range(b - a):
                        if is5_w[k]:
                            if it_w[k] in seen_items:
                                wt[r, k] = REWEIGHT
                            seen_items.add(it_w[k])
            it, ct, yy, ff, lm, ww = to_dev(item, cat, y, feats, lab_mask, wt, dev=dev)
            logit = model(it, ct, yy, ff, ff[..., 1:])
            w_eff = lm * ww
            loss = (lossf(logit, yy.float()) * w_eff).sum() / w_eff.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        auc_val, _ = eval_auc(model, data, dev, 1)
        print(f"{name} epoch {ep + 1} train loss {tot / max(1, len(order) // BATCH):.4f} val AUC {auc_val:.4f}", flush=True)
        if auc_val > best + 1e-4:
            best, bad = auc_val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    return model, best


def implied(model, name, data, dev, sample_idx):
    """p(actual last-200 context) - p(context without earlier tokens of the same item)."""
    model.eval()
    starts = np.repeat(data["starts"], data["ends"] - data["starts"])  # sequence start for every token
    act_lists, cf_lists = [], []
    for j in sample_idx:
        s = starts[j]
        ctx = np.arange(max(s, j - L + 1), j + 1)
        act_lists.append(ctx)
        lo = max(s, j - 5000)
        full = np.arange(lo, j + 1)
        keep = ~((data["item"][full] == data["item"][j]) & (full < j))
        cf_lists.append(full[keep][-L:])
    out = {}
    for tag, lists in (("act", act_lists), ("cf", cf_lists)):
        res = np.empty(len(sample_idx), np.float32)
        with torch.no_grad():
            for b0 in range(0, len(lists), 2 * BATCH):
                lb = lists[b0:b0 + 2 * BATCH]
                wins = [(0, len(x)) for x in lb]
                item, cat, y, feats = batchify(data, wins, idx_lists=lb)
                if tag == "cf" and name == "DKT-time":
                    for r, x in enumerate(lb):  # target item memory -> never seen
                        feats[r, len(x) - 1, 1:] = 0.0
                it, ct, yy, ff = to_dev(item, cat, y, feats, dev=dev)
                p = torch.sigmoid(model(it, ct, yy, ff, ff[..., 1:])).cpu().numpy()
                for r, x in enumerate(lb):
                    res[b0 + r] = p[r, len(x) - 1]
        out[tag] = res
    return out["act"], out["cf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clean"); ap.add_argument("out")
    ap.add_argument("--models", default="DKT,DKT-time,SAKT"); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--dim", type=int, default=128); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2); ap.add_argument("--reweight", type=float, default=1.0)
    ap.add_argument("--tag", default="", help="suffix for output files of a tuning configuration")
    ap.add_argument("--implied-only", action="store_true", help="load saved weights; recompute predictions and implied effects")
    a = ap.parse_args()
    global L, STRIDE, BATCH, D, LR, DROPOUT, REWEIGHT
    L, STRIDE, BATCH, D, LR, DROPOUT, REWEIGHT = a.window, a.window // 2, a.batch, a.dim, a.lr, a.dropout, a.reweight
    sfx = ("" if L == 200 else f"_L{L}") + (f"_{a.tag}" if a.tag else "")
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = os.path.join(a.out, "deep_tokens.npz")  # produced by rq2_deep_data.py
    data = {k: (v.item() if v.ndim == 0 else v) for k, v in np.load(cache, allow_pickle=True).items()}
    print("tokens:", len(data["y"]), "round-5 tokens:", int(data["is5"].sum()), "learners:", len(data["starts"]), "device:", dev, flush=True)
    te5 = data["is5"] & (data["split"] == 2)
    seen_te = np.flatnonzero(te5 & (data["exposure"] > 1))
    rng = np.random.default_rng(SEED)
    sample_idx = np.sort(rng.choice(seen_te, size=min(N_IMPLIED, len(seen_te)), replace=False))
    lb_all = lag_bin(np.where(data["exposure"] > 1, data["gap_h"], -1))
    met, imp_rows = [], []
    for name in a.models.split(","):
        wpath = os.path.join(a.out, f"deep_{name}{sfx}.pt")
        if a.implied_only:
            model = (DKT(data["n_item_tok"], data["n_cat"], time=(name == "DKT-time")) if name.startswith("DKT")
                     else SAKT(data["n_item_tok"], data["n_cat"])).to(dev)
            model.load_state_dict(torch.load(wpath, map_location=dev))
            best_val, _ = eval_auc(model, data, dev, 1)
        else:
            model, best_val = train_model(name, data, dev, a.epochs)
            torch.save(model.state_dict(), wpath)
        auc_te, p = eval_auc(model, data, dev, 2)
        sel = te5 & ~np.isnan(p)
        pd.DataFrame({"uid": data["uid"][sel], "game_id": data["game_id"][sel], "question_id": data["question_id"][sel],
                      "y": data["y"][sel], "p": p[sel]}).to_parquet(os.path.join(a.out, f"rq2_rows_deep_{name}{sfx}.parquet"), index=False)
        m = dict(model=f"M7 {name}{sfx}", n=int(sel.sum()), **B.metrics(data["y"][sel].astype(float), p[sel].astype(float)), val_AUC=best_val,
                 dim=D, lr=LR, dropout=DROPOUT, window=L, reweight=REWEIGHT)
        met.append(m)
        p_act, p_cf = implied(model, name, data, dev, sample_idx)
        imp = p_act - p_cf
        lb, prev = lb_all[sample_idx], data["prev_y"][sample_idx]
        pd.DataFrame({"uid": data["uid"][sample_idx], "game_id": data["game_id"][sample_idx],
                      "question_id": data["question_id"][sample_idx], "lag_bin": lb, "prev_y": prev, "implied": imp,
                      "p_act": p_act, "p_cf": p_cf}
                     ).to_parquet(os.path.join(a.out, f"rq2_implied_rows_deep_{name}{sfx}.parquet"), index=False)
        for lab, msk in [("all", np.ones(len(sample_idx), bool))] + [(f"lag {b}", lb == b) for _, _, b in LAG_BINS] + \
                        [(f"after correct, lag {b}", (lb == b) & (prev == 1)) for _, _, b in LAG_BINS] + \
                        [(f"after error, lag {b}", (lb == b) & (prev == 0)) for _, _, b in LAG_BINS]:
            imp_rows.append(dict(model=f"M7 {name}{sfx}", subset=lab, n=int(msk.sum()),
                                 implied_effect=float(imp[msk].mean()) if msk.any() else np.nan, item_specific_path=True))
        print(m, "implied all:", round(float(imp.mean()), 4), flush=True)
        for fname, rows in ((f"rq2_metrics_test_deep{sfx}.csv", met), (f"rq2_implied_effects_deep{sfx}.csv", imp_rows)):
            path, new_df = os.path.join(a.out, fname), pd.DataFrame(rows)
            if os.path.exists(path):  # keep rows of models not re-run now
                old_df = pd.read_csv(path)
                new_df = pd.concat([old_df[~old_df.model.isin(new_df.model.unique())], new_df], ignore_index=True)
            new_df.to_csv(path, index=False, float_format="%.5f")


if __name__ == "__main__":
    main()
