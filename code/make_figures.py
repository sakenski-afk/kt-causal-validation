"""
Publication figures for the RQ1-RQ3 paper (MDPI Applied Sciences): Times New Roman, 1200 dpi.

    python make_figures.py <work_dir> [out_dir]

<work_dir> = the working directory (design_checks, rq1, rq2 results). Writes Figure1..Figure8 as TIFF (LZW) and
PNG at 1200 dpi, width 16 cm (full text width) unless stated. Only aggregated results are plotted.
Colour: categorical slots 1-3 of the reference palette (blue, orange, aqua; validated all-pairs incl.
CVD) for the three models with an item-specific path; black = causal reference; grey = models without
an item path. Marker shapes duplicate the colour code for greyscale printing.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

CM = 1 / 2.54
W_FULL = 16 * CM
DPI = 1200
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
MODEL_STYLE = {  # colour, marker
    "M5 GBM (all features)": (BLUE, "o", "GBM"),
    "M6b DAS3H + item windows": (ORANGE, "s", "DAS3H + item windows"),
    "M7 DKT-time": (AQUA, "^", "DKT-time"),
}
LAGS = ["<1h", "1-6h", "6-24h", "1-3d", "3-7d", "7-14d", "14-30d", "30-90d", ">90d"]
LAG_LABELS = ["<1 h", "1–6 h", "6–24 h", "1–3 d", "3–7 d", "7–14 d", "14–30 d", "30–90 d", ">90 d"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"], "mathtext.fontset": "stix",
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.2,
    "savefig.dpi": DPI, "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
})


def save(fig, out, name):
    # journals expect RGB without an alpha channel: render once, flatten on white, write TIFF (LZW) + PNG
    from PIL import Image
    png = os.path.join(out, name + ".png")
    fig.savefig(png, dpi=DPI, facecolor="white", transparent=False)
    plt.close(fig)
    im = Image.open(png).convert("RGB")
    im.save(png, dpi=(DPI, DPI))
    im.save(os.path.join(out, name + ".tif"), dpi=(DPI, DPI), compression="tiff_lzw")
    print("saved", name, flush=True)


def panel_label(ax, s):
    ax.text(-0.13, 1.04, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left", color=INK)


# ------------------------------------------------------------------------------------------------
def fig1(work, out):
    """Methodological framework: a causal track (natural experiment) and a model track (knowledge tracing)
    built from the same logs and converging on the causal-consistency comparison. Schematic, no data; black and white."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(W_FULL, 12.5 * CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 160); ax.set_ylim(0, 125)

    def box(ax, x, y, w, h, fc="white", ec=INK2, lw=0.7, r=1.5, ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls))

    def arrow(ax, p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK2, shrinkA=0, shrinkB=0, zorder=3))

    def chip(ax, x, y, text, color, ha="center"):
        ax.text(x, y, text, ha=ha, va="center", fontsize=6.8, color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.6", fc="white", ec=color, lw=0.6), zorder=5)
    # lanes
    box(ax, 3, 30, 75, 72, fc="white", ec=MUTED, lw=0.6, r=2.5, ls=(0, (4, 2)))
    box(ax, 82, 30, 75, 72, fc="white", ec=MUTED, lw=0.6, r=2.5, ls=(0, (4, 2)))
    ax.text(40.5, 96, "CAUSAL TRACK  (natural experiment)", ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold")
    ax.text(119.5, 96, "MODEL TRACK  (knowledge tracing)", ha="center", va="center", fontsize=7.5, color=INK, fontweight="bold")
    # top: data
    box(ax, 22, 108, 116, 14)
    ax.text(80, 117.5, "Platform logs  →  analysis sample", ha="center", va="center", fontsize=8.5, fontweight="bold")
    ax.text(80, 111.8, "2.57 M round-5 responses · 18,484 learners · Jan 2025 – Mar 2026 · rules P1–P6", ha="center", va="center", fontsize=7, color=INK2)
    arrow(ax, (40.5, 108), (40.5, 102)); arrow(ax, (119.5, 108), (119.5, 102))

    def stage(x, y, title, lines, chipt=None, chipc=None, w=65, h=26):
        box(ax, x, y, w, h)
        ax.text(x + 2.5, y + h - 4.5, title, ha="left", va="center", fontsize=8, fontweight="bold")
        for k, l in enumerate(lines):
            ax.text(x + 2.5, y + h - 11 - 5 * k, l, ha="left", va="center", fontsize=7)
        if chipt:
            chip(ax, x + w - 2, y + h - 4.5, chipt, chipc, ha="right")
    stage(8, 64, "Design verification", ["Leave-learner-out expected draws (Eq. 1)", "Assumptions A1–A3 tested", "Global pool rotation → control"])
    stage(8, 34, "Causal reference  τ", ["Learner × game + item FE (Eq. 2)", "By lag and by previous outcome", "Per test window and whole period"], "RQ1 · H1", INK)
    stage(87, 64, "Fourteen KT models", ["Five families, M0–M7", "Rolling-origin windows F1–F3", "AUC, log-loss, RMSE, ECE (Eq. 3)"])
    stage(87, 34, "Implied effect  eₘ", ["Counterfactual query: history", "without the item (Eq. 4)", "By lag and by previous outcome"])
    arrow(ax, (40.5, 64), (40.5, 60)); arrow(ax, (119.5, 64), (119.5, 60))
    # convergence
    box(ax, 15, 3, 130, 21, fc="white", ec=INK, lw=0.9)
    ax.text(19, 19, "Causal consistency   Dₘ = ‖ eₘ − τ ‖  (Eq. 5)", ha="left", va="center", fontsize=8.5, fontweight="bold")
    ax.text(19, 12.8, "compared with predictive accuracy; by previous outcome, two scales, two bin weightings", ha="left", va="center", fontsize=7, color=INK2)
    ax.text(19, 7.0, "→ review policy: model half-life rule vs causal half-life (Eq. 6)", ha="left", va="center", fontsize=7, color=INK2, style="italic")
    chip(ax, 142, 19, "RQ2 · H2", INK, ha="right"); chip(ax, 142, 7.0, "RQ3 · H3", INK, ha="right")
    arrow(ax, (40.5, 34), (40.5, 24)); arrow(ax, (119.5, 34), (119.5, 24))
    save(fig, out, "Figure1_methodology")
    # strictly black and white: drop the warm tint of the palette greys
    from PIL import Image
    for ext, kw in (("png", {}), ("tif", {"compression": "tiff_lzw"})):
        f = os.path.join(out, "Figure1_methodology." + ext)
        Image.open(f).convert("L").save(f, dpi=(DPI, DPI), **kw)



def fig2(work, out):
    """Causal DAG of the design (black and white). Solid box = observed; dashed = unobserved; double border = conditioned
    on (learner x game fixed effect, item fixed effect, pool and global-recency controls). The identifying restriction
    is the ABSENCE of arrows from the learner's state and history into the draw (assumption A1)."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    fig = plt.figure(figsize=(W_FULL, 8.6 * CM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 160); ax.set_ylim(0, 86)

    def node(x, y, w, h, title, sub, latent=False, cond=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="white", ec=INK, lw=0.8, ls=(0, (3, 2)) if latent else "-", zorder=2))
        if cond:
            ax.add_patch(FancyBboxPatch((x - w / 2 - 1.2, y - h / 2 - 1.2), w + 2.4, h + 2.4,
                                        boxstyle="round,pad=0,rounding_size=1.8", fc="none", ec=INK, lw=0.6, zorder=2))
        ax.text(x, y + 2.3, title, ha="center", va="center", fontsize=8, color=INK, zorder=3)
        ax.text(x, y - 2.6, sub, ha="center", va="center", fontsize=7, color=INK2, zorder=3)

    def arr(p, q, ls="-", label=None, lx=0, ly=0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=INK, ls=ls,
                                     shrinkA=0, shrinkB=0, zorder=1))
        if label:
            ax.text(lx, ly, label, fontsize=7.5, ha="center", va="center", color=INK, zorder=4,
                    bbox=dict(fc="white", ec="none", pad=0.4))

    W, H = 34, 11
    node(20, 74, W, H, r"Learner history $H_u$", "categories, activity, return times")
    node(20, 46, W, H, r"Learner state $U_{ug}$", r"absorbed by $\alpha_{ug}$", latent=True, cond=True)
    node(78, 74, 38, H, r"Pool and rotation $(f_i, r_i)$", "category × language × month", cond=True)
    node(78, 46, W, H, r"Item drawn $D_{ugi}$", "random within the pool")
    node(78, 16, W, H, r"Recorded encounter", r"seen$_{ugi}$ and its lag")
    node(118, 46, 32, H, "Unrecorded encounters", r"rounds 1–4; depend on $H_u$", latent=True)
    node(143, 74, 28, H, r"Item difficulty $\delta_i$", "item fixed effect", cond=True)
    node(140, 16, 30, H, r"Answer $y_{ugi}$", "correct / wrong")
    node(26, 16, 36, H, "Return time, category choice", "not randomized: sets the lag", latent=False)
    arr((20, 68.5), (20, 52.7))                                   # H -> U
    arr((20, 39.3), (20, 21.5))                                   # U -> return time
    arr((44, 16), (61, 16), label="lag", lx=52.5, ly=19)          # return time -> lag of the recorded encounter
    ax.text(80, 36.5, "item selection: randomized", fontsize=6.8, ha="left", va="center", color=INK2, style="italic")
    arr((37, 71), (66, 21.5))                                     # H -> recorded encounter (which items u has met)
    arr((78, 68.5), (78, 51.5))                                   # pool -> draw
    arr((78, 40.5), (78, 21.5))                                   # draw -> recorded encounter
    arr((95, 16), (125, 16), label=r"$\beta$  (target)", lx=110, ly=19.5)
    arr((37, 44), (125, 20.5))                                    # U -> y
    arr((97, 71), (106, 51.5), ls=(0, (3, 2)))                    # pool -> unrecorded (random draws in rounds 1-4)
    arr((124, 40.5), (134, 21.5), ls=(0, (3, 2)))                 # unrecorded -> y
    arr((148, 68.3), (148, 21.5))                                 # difficulty -> y
    # arrows ruled out by the design (A1), drawn as crossed dotted lines
    for p, q, tx in (((37, 76), (59, 50), (49, 61.5)), ((37, 47.5), (61, 47.5), (57.5, 47.5))):
        ax.plot([p[0], q[0]], [p[1], q[1]], color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=0)
        ax.text(tx[0], tx[1], "×", ha="center", va="center", fontsize=11, color=INK, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.05), zorder=4)
    ax.text(59.5, 57.5, "no arrow (A1)", fontsize=7, ha="left", color=INK, style="italic")
    ax.text(3, 3, "solid box: observed · dashed box: unobserved · double border: conditioned on · "
            "×: arrow ruled out by the design and tested in the data", fontsize=6.8, ha="left", va="center", color=INK2)
    save(fig, out, "Figure2_causal_graph")
    from PIL import Image
    for ext, kw in (("png", {}), ("tif", {"compression": "tiff_lzw"})):
        f = os.path.join(out, "Figure2_causal_graph." + ext)
        Image.open(f).convert("L").save(f, dpi=(DPI, DPI), **kw)


def fig3(work, out):
    """Design check: (a) draw ratio by global recency; (b) balance by quarter."""
    deck = pd.read_csv(os.path.join(work, "design_checks", "deck_global_recency.csv"))
    order = ["<10min", "10-60min", "1-6h", "6-24h", "1-7d", ">7d", "never"]
    labels = ["<10 min", "10–60 min", "1–6 h", "6–24 h", "1–7 d", ">7 d", "never\ndealt"]
    bal = pd.read_csv(os.path.join(work, "design_checks", "randomization_balance.csv"))
    bal = bal[(bal.window == "month") & bal.subset.str.startswith("quarter") & bal.property.isin(["seen", "prev_wrong"])]
    bal = bal[bal.subset.str[8:12].astype(int) * 10 + bal.subset.str[-1].astype(int) <= 20261]  # analysis period
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 6.2 * CM), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
    ax = axes[0]
    x = np.arange(len(order))
    for k, (status, col, mk, lab) in enumerate((("unseen", BLUE, "o", "Items not seen by the learner"),
                                                 ("seen", ORANGE, "s", "Items seen by the learner"))):
        d = deck[deck.learner_status == status].set_index("global_recency").reindex(order)
        xx = x + (k - 0.5) * 0.18
        ok = d.ratio.notna().values
        ax.errorbar(xx[ok], d.ratio.values[ok], yerr=[(d.ratio - d.ci_low).values[ok], (d.ci_high - d.ratio).values[ok]],
                    fmt=mk, color=col, ms=4, mec="white", mew=0.6, elinewidth=0.8, capsize=0, label=lab, zorder=3)
    ax.axhline(1, color=INK2, lw=0.7, zorder=2)
    ax.set_xticks(x, labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_xlabel("Time since the item was last dealt in any game")
    ax.set_ylabel("Observed / expected draws")
    ax.set_ylim(0.4, 1.6)
    ax.legend(loc="upper left", handletextpad=0.3)
    panel_label(ax, "(a)")
    ax = axes[1]
    quarters = sorted(bal.subset.unique())
    qx = np.arange(len(quarters))
    for k, (prop, col, mk, lab) in enumerate((("seen", BLUE, "o", "Previously seen item"),
                                              ("prev_wrong", ORANGE, "s", "Previously answered wrongly"))):
        d = bal[bal.property == prop].set_index("subset").reindex(quarters)
        rel, lo, hi = 100 * d["diff"] / d.expected, 100 * d.ci_low / d.expected, 100 * d.ci_high / d.expected
        xx = qx + (k - 0.5) * 0.18
        ax.errorbar(xx, rel, yerr=[rel - lo, hi - rel], fmt=mk, color=col, ms=4, mec="white", mew=0.6,
                    elinewidth=0.8, capsize=0, label=lab, zorder=3)
    ax.axhline(0, color=INK2, lw=0.7, zorder=2)
    ax.set_xticks(qx, [q.replace("quarter ", "").replace("Q", " Q") for q in quarters])
    ax.set_xlabel("Quarter")
    ax.set_ylabel("Observed − expected (% of expected)")
    ax.set_ylim(-12, 12)
    ax.legend(loc="upper left", handletextpad=0.3)
    panel_label(ax, "(b)")
    save(fig, out, "Figure3_design_check")


# ------------------------------------------------------------------------------------------------
def fig4(work, out):
    """Causal re-exposure effect by lag: LPM (E1) vs conditional logit; after correct vs after error."""
    e = pd.read_csv(os.path.join(work, "rq1", "rq1_effects.csv"))
    e1 = e[e.estimator.str.startswith("E1")]
    b = e1[e1.spec == "B lag"].set_index("term").loc[["lag " + l for l in LAGS]]
    cl = pd.read_csv(os.path.join(work, "rq1", "rq1d_clogit.csv"))
    cl = cl[cl.spec == "B lag"].set_index("term").loc[["lag " + l for l in LAGS]]
    c = e1[e1.spec == "C lag x prev"].set_index("term")
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 6.4 * CM), sharex=True, gridspec_kw=dict(wspace=0.28))
    x = np.arange(len(LAGS))
    ax = axes[0]
    # 95% CIs as error bars (no shaded bands); the two series are offset slightly so that the bars do not overlap
    ax.errorbar(x - 0.1, 100 * b.coef, yerr=[100 * (b.coef - b.ci_low), 100 * (b.ci_high - b.coef)], color=INK, marker="o", ms=3.5,
                mec="white", mew=0.5, elinewidth=0.8, capsize=1.5, label="Linear probability model (E1)")
    ax.errorbar(x + 0.1, cl.pp_at_054, yerr=[cl.pp_at_054 - cl.pp_lo, cl.pp_hi - cl.pp_at_054], color=BLUE, marker="s", ms=3.5,
                mec="white", mew=0.5, elinewidth=0.8, capsize=1.5, label="Conditional logit (at p = 0.54)")
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_ylabel("Effect of a previous encounter (pp)")
    ax.set_xlabel("Time since the previous encounter")
    ax.set_ylim(-2, 25)
    ax.legend(loc="upper right", handletextpad=0.4)
    panel_label(ax, "(a)")
    ax = axes[1]
    for lab, col, mk, name in (("after correct", BLUE, "o", "Previous answer correct"),
                               ("after error", ORANGE, "s", "Previous answer wrong")):
        d = c.loc[[f"{lab}, lag {l}" for l in LAGS]]
        off = -0.1 if lab == "after correct" else 0.1
        ax.errorbar(x + off, 100 * d.coef, yerr=[100 * (d.coef - d.ci_low), 100 * (d.ci_high - d.coef)], color=col, marker=mk, ms=3.5,
                    mec="white", mew=0.5, elinewidth=0.8, capsize=1.5, label=name)
    ax.axhline(0, color=INK2, lw=0.7)
    ax.set_xlabel("Time since the previous encounter")
    ax.set_ylabel("Conditional contrast (pp)")
    ax.legend(loc="upper right", handletextpad=0.4)
    panel_label(ax, "(b)")
    for ax in axes:
        ax.set_xticks(x, LAG_LABELS, rotation=45, ha="right", rotation_mode="anchor")
    save(fig, out, "Figure4_causal_curves")


# ------------------------------------------------------------------------------------------------
def fig5(work, out):
    """Predictive accuracy vs causal consistency: (a) all models, test window F3; (b) item-memory models pooled over 3 folds.
    All labels are in a legend below the panels, so nothing is drawn over the data."""
    # AKT and HawkesKT (MAKE revision): F3 bootstrap rerun with them (deep_modern/F3_boot, original rows unchanged) and
    # pooled levels from review_modern_kt.py
    f3_boot = os.path.join(work, "deep_modern", "F3_boot", "rq2_bootstrap_summary.csv")
    s = pd.read_csv(f3_boot if os.path.exists(f3_boot) else os.path.join(work, "rq2", "rq2_bootstrap_summary.csv"))
    # M0 (AUC 0.50) and M1 item difficulty (AUC 0.69) lie outside the plotted AUC range (stated in the text)
    s = s[~s.model.str.endswith("_L500") & ~s.model.isin(["M0 Global mean", "M1 Item difficulty"])]
    pooled = pd.read_csv(os.path.join(work, "rq2", "rq2_folds_summary.csv"))
    lev_path = os.path.join(work, "deep_modern", "modern_levels.csv")
    if os.path.exists(lev_path):
        lev = pd.read_csv(lev_path)
        lev = lev[(lev.group == "all") & lev.model.isin(["M7 AKT", "M7 HawkesKT"])]
        pooled = pd.concat([pooled, pd.DataFrame({"model": lev.model, "AUC": lev.AUC, "AUC_lo": lev.AUC_lo, "AUC_hi": lev.AUC_hi,
                                                  "rmse_lag": lev.D_pp / 100, "rmse_lo": lev.lo / 100, "rmse_hi": lev.hi / 100})],
                           ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 8.4 * CM), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=0.3))
    styles = dict(MODEL_STYLE)
    styles["M5a GBM without memory features"] = (GREY, "D", "GBM w/o memory features")
    OTHER = ("#B5B5AE", "o", "Models implying no item-specific effect (PFA, Elo, DAS3H, DKT, SAKT, AKT, HawkesKT, IRT-1PL)")
    for ax, data, tag, title in ((axes[0], s, "(a)", "Test window Jan–Mar 2026, all models"),
                                 (axes[1], pooled, "(b)", "Three test windows pooled")):
        for _, r in data.iterrows():
            col, mk, lab = styles.get(r.model, OTHER)
            big = r.model in MODEL_STYLE
            ax.errorbar(r.AUC, 100 * r.rmse_lag, xerr=[[r.AUC - r.AUC_lo], [r.AUC_hi - r.AUC]],
                        yerr=[[100 * (r.rmse_lag - r.rmse_lo)], [100 * (r.rmse_hi - r.rmse_lag)]],
                        fmt=mk, color=col, ms=5 if big else 3.5, mec="white", mew=0.6, elinewidth=0.7, capsize=0,
                        alpha=1 if r.model in styles else 0.8, zorder=3 if big else 2)
        ax.set_xlabel("Test AUC")
        ax.set_ylim(0, 3.7)
        ax.set_title(title, loc="left", color=INK)
        panel_label(ax, tag)
    axes[0].set_ylabel("Causal inconsistency (RMSE, pp)")
    axes[0].set_xlim(0.755, 0.822)
    axes[1].set_xlim(0.755, 0.830)
    handles = [Line2D([], [], marker=mk, color=col, ls="", ms=5, mec="white", label=lab) for col, mk, lab in MODEL_STYLE.values()]
    handles.append(Line2D([], [], marker="D", color=GREY, ls="", ms=4, mec="white", label=styles["M5a GBM without memory features"][2]))
    handles.append(Line2D([], [], marker=OTHER[1], color=OTHER[0], ls="", ms=4, mec="white", label=OTHER[2]))
    fig.subplots_adjust(bottom=0.27)
    fig.legend(handles=handles[:4], loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, handletextpad=0.1, columnspacing=1.4)
    fig.legend(handles=handles[4:], loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=1, handletextpad=0.1)
    save(fig, out, "Figure5_auc_vs_consistency")


# ------------------------------------------------------------------------------------------------
def fig6(work, out):
    """Pooled (3 folds) model-implied vs causal effects by lag, after a correct and after a wrong answer."""
    c = pd.read_csv(os.path.join(work, "rq2", "rq2_folds_curves.csv"))
    x = np.arange(len(LAGS))
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 8.0 * CM), sharex=True, gridspec_kw=dict(wspace=0.25))
    for ax, (grp, title, tag) in zip(axes, (("after correct", "Previous answer correct", "(a)"),
                                            ("after error", "Previous answer wrong", "(b)"))):
        g = c[c.group == grp]
        ref = g[g.model == "CAUSAL"].set_index("lag_bin").loc[LAGS]
        ax.errorbar(x, 100 * ref.effect, yerr=[100 * (ref.effect - ref.ci_low), 100 * (ref.ci_high - ref.effect)], color=INK, lw=1.4,
                    marker="o", ms=3.5, mec="white", mew=0.5, elinewidth=0.8, capsize=1.5, label="Causal reference (95% CI)", zorder=4)
        for m, (col, mk, lab) in MODEL_STYLE.items():
            d = g[g.model == m].set_index("lag_bin").loc[LAGS]
            ax.plot(x, 100 * d.effect, color=col, marker=mk, ms=3.5, mec="white", mew=0.5, label=lab)
        ax.axhline(0, color=INK2, lw=0.7)
        ax.set_xticks(x, LAG_LABELS, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Time since the previous encounter")
        ax.set_title(title, loc="left", color=INK)
        panel_label(ax, tag)
    axes[0].set_ylabel("Effect of a previous encounter (pp)")
    h, l = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(bottom=0.30)
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.4)
    save(fig, out, "Figure6_model_vs_causal")


# ------------------------------------------------------------------------------------------------
def fig8(work, out):
    """Review policy (whole-period causal curve, fine lag grid): model half-lives from three folds vs causal half-life."""
    paths = {"F1": os.path.join(work, "rq2", "fold_2025-07", "rq3_policy_fine.csv"),
             "F2": os.path.join(work, "rq2", "fold_2025-10", "rq3_policy_fine.csv"),
             "F3": os.path.join(work, "rq2", "rq3_policy_fine.csv")}
    pol = {k: pd.read_csv(v).set_index(["group", "model"]) for k, v in paths.items()}
    rows = [("M5 GBM (all features)",) + MODEL_STYLE["M5 GBM (all features)"],
            ("M6b DAS3H + item windows",) + MODEL_STYLE["M6b DAS3H + item windows"],
            ("M7 DKT-time",) + MODEL_STYLE["M7 DKT-time"]]
    fig, axes = plt.subplots(1, 2, figsize=(W_FULL, 5.4 * CM), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, (grp, title, tag) in zip(axes, (("after correct", "After a correct answer", "(a)"),
                                            ("after error", "After a wrong answer", "(b)"))):
        ref = pol["F3"].loc[(grp, "CAUSAL (reference)")]
        for v in (ref.half_life_lo, ref.half_life_hi):  # 95% interval of the causal half-life as thin dashed lines (no shading)
            ax.axvline(v, color=INK2, lw=0.6, ls=(0, (3, 2)), zorder=1)
        ax.errorbar(ref.half_life_h, len(rows), xerr=[[ref.half_life_h - ref.half_life_lo], [ref.half_life_hi - ref.half_life_h]],
                    fmt="o", color=INK, ms=5, mec="white", mew=0.6, elinewidth=0.8, capsize=0, zorder=3)
        for k, (m, col, mk, lab) in enumerate(rows):
            yk = len(rows) - 1 - k
            hs = [pol[f].loc[(grp, m)].half_life_h for f in ("F1", "F2", "F3")]
            rets = [pol[f].loc[(grp, m)].retained_at_review for f in ("F1", "F2", "F3")]
            ax.plot(hs, [yk + d for d in (-0.18, 0, 0.18)], mk, color=col, ms=4.5, mec="white", mew=0.5, ls="", zorder=3)
            ax.annotate(f"{min(rets) * 100:.0f}–{max(rets) * 100:.0f}% retained", (max(hs), yk), xytext=(7, 0),
                        textcoords="offset points", va="center", fontsize=7.5, color=INK2)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 5000)
        ax.set_xticks([1, 10, 100, 1000], ["1", "10", "100", "1000"])
        ax.set_xlabel("Review interval = half-life (hours, log scale)")
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="y", visible=False)
        panel_label(ax, tag)
    axes[0].set_yticks(range(len(rows) + 1), [r[3] for r in rows][::-1] + ["Causal reference"])
    save(fig, out, "Figure8_review_policy")


def fig7(work, out):
    """Predictive quality (test window F3): calibration residuals (observed - predicted) on all responses and on
    re-exposures; log-loss of all models with learner-clustered 95% CIs."""
    cal = pd.read_csv(os.path.join(work, "rq2", "rq2_calibration_bins.csv"))
    met = pd.read_csv(os.path.join(work, "rq2", "rq2_metrics_full_F3.csv"))
    short = {"M2 IRT-1PL": "IRT-1PL", "M3 PFA": "PFA", "M4 Elo (online)": "Elo", "M6 DAS3H": "DAS3H",
             "M7 DKT": "DKT", "M7 SAKT": "SAKT", "M5a GBM without memory features": "GBM w/o memory",
             "M5 GBM (all features)": "GBM", "M6b DAS3H + item windows": "DAS3H + windows", "M7 DKT-time": "DKT-time"}
    shown = {m: (c, mk) for m, (c, mk, _) in MODEL_STYLE.items()}
    shown["M5a GBM without memory features"] = (GREY, "D")
    fig, axes = plt.subplots(1, 3, figsize=(W_FULL, 7.0 * CM), gridspec_kw=dict(width_ratios=[1, 1, 1.0], wspace=0.75))
    for ax, subset, title, tag in ((axes[0], "all", "All test responses", "(a)"), (axes[1], "re-exposures", "Re-exposures only", "(b)")):
        ax.axhline(0, color=INK2, lw=0.7, zorder=1)
        for m, (col, mk) in shown.items():
            d = cal[(cal.model == m) & (cal.subset == subset)].sort_values("mean_pred")
            ax.plot(d.mean_pred, d.obs - d.mean_pred, color=col, marker=mk, ms=2.8, mec="white", mew=0.4, lw=1.0,
                    label=short[m], zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(-0.06, 0.16)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xlabel("Predicted P(correct)")
        ax.set_title(title, loc="left", color=INK)
        ax.text(-0.03, 1.04, tag, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right", color=INK)
    axes[0].set_ylabel("Observed − predicted")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=7.5, frameon=False, bbox_to_anchor=(0.36, 0.0), handletextpad=0.3)
    fig.subplots_adjust(bottom=0.26)
    ax = axes[2]
    met = met[met.model != "M1 Item difficulty"].sort_values("LogLoss", ascending=False).reset_index(drop=True)
    for i, r in met.iterrows():
        col, mk = shown.get(r.model, (GREY, "D"))
        if r.model not in MODEL_STYLE:
            col = GREY if r.model == "M5a GBM without memory features" else "#B5B5AE"
        ax.errorbar(r.LogLoss, i, xerr=[[r.LogLoss - r.LogLoss_lo], [r.LogLoss_hi - r.LogLoss]], fmt=mk, color=col,
                    ms=4, mec="white", mew=0.5, elinewidth=0.8, capsize=0)
    ax.set_yticks(range(len(met)))
    ax.set_yticklabels([short.get(m, m) for m in met.model], fontsize=7.5)
    ax.set_xlabel("Test log-loss (lower = better)")
    ax.set_title("Log-loss, 95% CI", loc="left", color=INK)
    ax.text(-0.03, 1.04, "(c)", transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right", color=INK)
    ax.grid(axis="y", visible=False)
    save(fig, out, "Figure7_predictive_quality")


def main(work, out=None):
    out = out or os.path.join(work, "paper", "figures")
    os.makedirs(out, exist_ok=True)
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        f(work, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
