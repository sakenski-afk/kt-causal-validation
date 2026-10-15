"""
RQ2 extension (MAKE revision): two recent knowledge-tracing architectures with built-in forgetting, evaluated with the
same token stream, training protocol and counterfactual queries as DKT / DKT-time / SAKT (rq2_deep.py).

    python rq2_deep_modern.py <token_dir> <out_dir> [--models AKT,HawkesKT] [--epochs 12]

<token_dir> holds deep_tokens.npz (rq2_deep_data.py); weights, test predictions, metrics and implied effects
are written to <out_dir> in the stage formats (rq2_rows_deep_<m>.parquet, rq2_metrics_test_deep.csv,
rq2_implied_rows_deep_<m>.parquet [direct protocol], rq2_implied_rows_deep_<m>_full.parquet [full deletion],
rq2_implied_effects_deep.csv).

Shared settings (unchanged from rq2_deep.py): d = 128, seed 42, Adam 1e-3, batch 256 windows of 200 tokens,
dropout 0.2, <= 12 epochs, early stopping on validation AUC (patience 2), round-5 tokens scored only.

  AKT       Ghosh, Heffernan & Lan (KDD 2020), as in the pyKT implementation: Rasch embeddings (category embedding +
            scalar item difficulty x category variation vector, L2 1e-5 on difficulties), question and interaction
            encoders and a knowledge retriever, 8 heads, monotonic attention (exponential decay with context-aware
            distance, learned per head). Distance is in positions, not clock time, as in the original.
  HawkesKT  Wang et al. (WSDM 2021): logit = item bias + category bias + sum over earlier interactions of
            alpha(source interaction, target) * exp(-beta(source interaction, target) * log(1 + dt_seconds)/log 5),
            alpha and beta from inner products of embeddings, beta clamped to [0, 10]. Sources and targets are item
            tokens (round-5 items; category placeholders for rounds 1-4), so item-specific excitation is possible.
            log(1 + dt) replaces log(dt + 1e-10) of the original for numerical stability at dt = 0.

The absolute token time (hours since the learner's first token) is put into feature channel 0, which only DKT-time
uses (as the time since the previous token); the counterfactual masking of review_cf_deep.run zeroes channels
1..3 only, so clock times of masked tokens are kept, as the direct protocol requires.
"""
import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baselines as B  # noqa: E402
import rq2_deep as Dm  # noqa: E402
from review_cf_deep import run as cf_run  # noqa: E402
from rq2_models import LAG_BINS, lag_bin  # noqa: E402

D, DROPOUT, HEADS, L2_RASCH = Dm.D, Dm.DROPOUT, 8, 1e-5


# ----------------------------------------------------------------------------------------------
class MonoAttention(nn.Module):
    """AKT multi-head attention with monotonic (distance-decayed) scores; kq_same as in AKT."""

    def __init__(self, kq_same):
        super().__init__()
        self.dk, self.h = D // HEADS, HEADS
        self.q = nn.Linear(D, D)
        self.k = self.q if kq_same else nn.Linear(D, D)
        self.v, self.o = nn.Linear(D, D), nn.Linear(D, D)
        self.gamma = nn.Parameter(torch.zeros(HEADS, 1, 1))
        nn.init.xavier_uniform_(self.gamma)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, q, k, v, strict):
        bs, T, _ = q.shape
        sp = lambda x: x.view(bs, T, self.h, self.dk).transpose(1, 2)
        Q, K, V = sp(self.q(q)), sp(self.k(k)), sp(self.v(v))
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=q.device), -1 if strict else 0)
        pos = torch.arange(T, device=q.device)
        with torch.no_grad():
            s_ = F.softmax(scores.masked_fill(~mask, -1e32), -1) * mask
            dist = torch.clamp((s_.sum(-1, keepdim=True) - s_.cumsum(-1)) * (pos[None, :] - pos[:, None]).abs(), min=0.).sqrt()
        g = -F.softplus(self.gamma)[None]
        scores = scores * torch.clamp((dist * g).exp(), 1e-5, 1e5)
        att = F.softmax(scores.masked_fill(~mask, -1e32), -1)
        if strict:  # first position has no past
            att = torch.cat([torch.zeros_like(att[:, :, :1]), att[:, :, 1:]], 2)
        out = (self.drop(att) @ V).transpose(1, 2).reshape(bs, T, D)
        return self.o(out)


class AKTLayer(nn.Module):
    def __init__(self, kq_same, ffn):
        super().__init__()
        self.att, self.ffn_on = MonoAttention(kq_same), ffn
        self.ln1, self.d1 = nn.LayerNorm(D), nn.Dropout(DROPOUT)
        if ffn:
            self.ffn = nn.Sequential(nn.Linear(D, 2 * D), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(2 * D, D))
            self.ln2, self.d2 = nn.LayerNorm(D), nn.Dropout(DROPOUT)

    def forward(self, q, k, v, strict):
        x = self.ln1(q + self.d1(self.att(q, k, v, strict)))
        if self.ffn_on:
            x = self.ln2(x + self.d2(self.ffn(x)))
        return x


class AKT(nn.Module):
    def __init__(self, n_item, n_cat):
        super().__init__()
        self.n_cat = n_cat
        self.c, self.c_diff = nn.Embedding(n_cat, D), nn.Embedding(n_cat, D)
        self.cr, self.cr_diff = nn.Embedding(2, D), nn.Embedding(2, D)
        self.mu = nn.Embedding(n_item, 1, padding_idx=0)
        nn.init.zeros_(self.mu.weight)
        self.enc_y = AKTLayer(kq_same=True, ffn=True)    # interaction encoder (includes current)
        self.enc_x = AKTLayer(kq_same=True, ffn=False)   # question encoder (includes current)
        self.retr = AKTLayer(kq_same=True, ffn=True)     # knowledge retriever (strict past)
        self.out = nn.Sequential(nn.Linear(2 * D, 512), nn.ReLU(), nn.Dropout(DROPOUT),
                                 nn.Linear(512, 256), nn.ReLU(), nn.Dropout(DROPOUT), nn.Linear(256, 1))
        self.reg = torch.zeros(())

    def forward(self, item, cat, y, feats, tfeats):
        mu = self.mu(item)
        x = self.c(cat) + mu * self.c_diff(cat)
        yy = self.c(cat) + self.cr(y) + mu * (self.c_diff(cat) + self.cr_diff(y))
        self.reg = (mu ** 2).sum() * L2_RASCH
        hy = self.enc_y(yy, yy, yy, strict=False)
        hx = self.enc_x(x, x, x, strict=False)
        h = self.retr(hx, hx, hy, strict=True)
        return self.out(torch.cat([h, x], -1)).squeeze(-1)


class HawkesKT(nn.Module):
    def __init__(self, n_item, n_cat):
        super().__init__()
        self.a_src, self.a_tgt = nn.Embedding(2 * n_item, D), nn.Embedding(n_item, D)
        self.b_src, self.b_tgt = nn.Embedding(2 * n_item, D), nn.Embedding(n_item, D)
        self.base_i, self.base_c = nn.Embedding(n_item, 1), nn.Embedding(n_cat, 1)
        for e in (self.a_src, self.a_tgt, self.b_src, self.b_tgt, self.base_i, self.base_c):
            nn.init.normal_(e.weight, 0.0, 0.01)
        self.reg = torch.zeros(())

    def forward(self, item, cat, y, feats, tfeats):
        inter = item * 2 + y
        alpha = self.a_src(inter) @ self.a_tgt(item).transpose(-2, -1)          # [b, src, tgt]
        beta = torch.clamp(self.b_src(inter) @ self.b_tgt(item).transpose(-2, -1) + 1, 0, 10)
        t = feats[..., 0] * 3600.0                                               # seconds
        dt = torch.log1p((t[:, None, :] - t[:, :, None]).abs()) / math.log(5)   # [b, src, tgt]
        T = item.shape[1]
        past = torch.triu(torch.ones(T, T, dtype=torch.bool, device=item.device), 1)  # src < tgt
        eff = (alpha * torch.exp(-beta * dt)).masked_fill(~past, 0).sum(1)
        return self.base_i(item).squeeze(-1) + self.base_c(cat).squeeze(-1) + eff


FACTORY = {"AKT": AKT, "HawkesKT": HawkesKT}


# ----------------------------------------------------------------------------------------------
def train_model(name, data, dev, epochs):
    """Same loop as rq2_deep.train_model (plus AKT's Rasch L2 term)."""
    Dm.set_seed(Dm.SEED)
    model = FACTORY[name](data["n_item_tok"], data["n_cat"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=Dm.LR)
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    train_wins = Dm.windows(data["starts"], data["ends"], Dm.L, upto_mask=data["split"] == 0)
    best, best_state, bad = -1, None, 0
    rng = np.random.default_rng(Dm.SEED)
    for ep in range(epochs):
        model.train()
        order = rng.permutation(len(train_wins))
        tot = 0.0
        for b0 in range(0, len(order), Dm.BATCH):
            wb = [train_wins[k] for k in order[b0:b0 + Dm.BATCH]]
            item, cat, y, feats = Dm.batchify(data, wb)
            lab_mask = np.zeros(item.shape, bool)
            for r, (a, b) in enumerate(wb):
                lab_mask[r, :b - a] = data["is5"][a:b]
            lab_mask[:, 0] = False
            it, ct, yy, ff, lm = Dm.to_dev(item, cat, y, feats, lab_mask, dev=dev)
            logit = model(it, ct, yy, ff, ff[..., 1:])
            lmf = lm.float()
            loss = (lossf(logit, yy.float()) * lmf).sum() / lmf.sum().clamp(min=1) + model.reg
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        auc_val, _ = Dm.eval_auc(model, data, dev, 1)
        print(f"{name} epoch {ep + 1} train loss {tot / max(1, len(order) // Dm.BATCH):.4f} val AUC {auc_val:.4f}", flush=True)
        if auc_val > best + 1e-4:
            best, bad = auc_val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                break
    model.load_state_dict(best_state)
    return model, best


def summarize(name, imp, lb, prev):
    rows = []
    for lab, msk in [("all", np.ones(len(imp), bool))] + [(f"lag {b}", lb == b) for _, _, b in LAG_BINS] + \
                    [(f"after correct, lag {b}", (lb == b) & (prev == 1)) for _, _, b in LAG_BINS] + \
                    [(f"after error, lag {b}", (lb == b) & (prev == 0)) for _, _, b in LAG_BINS]:
        rows.append(dict(model=f"M7 {name}", subset=lab, n=int(msk.sum()),
                         implied_effect=float(imp[msk].mean()) if msk.any() else np.nan, item_specific_path=True))
    return rows


def upsert(path, rows):
    new_df = pd.DataFrame(rows)
    if os.path.exists(path):
        old = pd.read_csv(path)
        new_df = pd.concat([old[~old.model.isin(new_df.model.unique())], new_df], ignore_index=True)
    new_df.to_csv(path, index=False, float_format="%.5f")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens"); ap.add_argument("out")
    ap.add_argument("--models", default="AKT,HawkesKT"); ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--implied-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = {k: (v.item() if v.ndim == 0 else v) for k, v in np.load(os.path.join(a.tokens, "deep_tokens.npz"), allow_pickle=True).items()}
    # channel 0 -> absolute clock time in hours since the learner's first token
    dt = np.expm1(data["feats"][:, 0].astype(np.float64))
    seq_id = np.repeat(np.arange(len(data["starts"])), data["ends"] - data["starts"])
    cs = np.cumsum(dt)
    data["feats"] = data["feats"].copy()
    data["feats"][:, 0] = (cs - cs[data["starts"]][seq_id] + dt[data["starts"]][seq_id]).astype(np.float32)
    print("tokens:", len(data["y"]), "learners:", len(data["starts"]), "device:", dev, flush=True)
    te5 = data["is5"] & (data["split"] == 2)
    seen_te = np.flatnonzero(te5 & (data["exposure"] > 1))
    rng = np.random.default_rng(Dm.SEED)
    sample_idx = np.sort(rng.choice(seen_te, size=min(Dm.N_IMPLIED, len(seen_te)), replace=False))
    lb = lag_bin(np.where(data["exposure"] > 1, data["gap_h"], -1))[sample_idx]
    prev = data["prev_y"][sample_idx]
    for name in a.models.split(","):
        wpath = os.path.join(a.out, f"deep_{name}.pt")
        if a.implied_only:
            model = FACTORY[name](data["n_item_tok"], data["n_cat"]).to(dev)
            model.load_state_dict(torch.load(wpath, map_location=dev))
            best_val, _ = Dm.eval_auc(model, data, dev, 1)
        else:
            model, best_val = train_model(name, data, dev, a.epochs)
            torch.save(model.state_dict(), wpath)
        auc_te, p = Dm.eval_auc(model, data, dev, 2)
        sel = te5 & ~np.isnan(p)
        pd.DataFrame({"uid": data["uid"][sel], "game_id": data["game_id"][sel], "question_id": data["question_id"][sel],
                      "y": data["y"][sel], "p": p[sel]}).to_parquet(os.path.join(a.out, f"rq2_rows_deep_{name}.parquet"), index=False)
        m = dict(model=f"M7 {name}", n=int(sel.sum()), **B.metrics(data["y"][sel].astype(float), p[sel].astype(float)),
                 val_AUC=best_val, dim=D, lr=Dm.LR, dropout=DROPOUT, window=Dm.L, reweight=1.0)
        upsert(os.path.join(a.out, "rq2_metrics_test_deep.csv"), [m])
        model.eval()
        p_act = cf_run(model, "ACT", data, dev, sample_idx, "actual")
        for protocol, sfx in (("direct", ""), ("full", "_full")):
            p_cf = cf_run(model, name, data, dev, sample_idx, protocol)
            imp = p_act - p_cf
            pd.DataFrame({"uid": data["uid"][sample_idx], "game_id": data["game_id"][sample_idx],
                          "question_id": data["question_id"][sample_idx], "lag_bin": lb, "prev_y": prev, "implied": imp,
                          "p_act": p_act, "p_cf": p_cf}).to_parquet(os.path.join(a.out, f"rq2_implied_rows_deep_{name}{sfx}.parquet"), index=False)
            if protocol == "direct":
                upsert(os.path.join(a.out, "rq2_implied_effects_deep.csv"), summarize(name, imp, lb, prev))
            print(name, protocol, "implied all:", round(float(imp.mean()), 5), flush=True)
        print(m, flush=True)


if __name__ == "__main__":
    main()
