"""
Generate publication-quality figures for EMNLP 2026 submission.
Outputs: EMNLP2026_submission/figures/Figure{1-4}.pdf

Usage:
    conda run -n base python scripts/generate_emnlp_figures.py
"""

import json
import os
from pathlib import Path
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).parent.parent
OUT  = ROOT / "EMNLP2026_submission" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ── shared style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "legend.fontsize":  9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.05,
})

# ── data ────────────────────────────────────────────────────────────────────
MODELS_ORDERED = [
    "Llama-3.1-70B", "QwQ-32B", "Llama-3.1-8B",
    "DeepSeek-V3", "Qwen2.5-72B", "GPT-5.1", "Qwen2.5-7B",
]

BA_NONE = {
    "GPT-5.1":       0.584,
    "Qwen2.5-7B":    0.599,
    "Llama-3.1-8B":  0.510,
    "Qwen2.5-72B":   0.558,
    "DeepSeek-V3":   0.547,
    "QwQ-32B":       0.400,
    "Llama-3.1-70B": 0.336,
}

BA_USER = {
    "GPT-5.1":       0.613,
    "Qwen2.5-7B":    0.550,
    "Llama-3.1-8B":  0.494,
    "Qwen2.5-72B":   0.617,
    "DeepSeek-V3":   0.606,
    "QwQ-32B":       0.614,
    "Llama-3.1-70B": 0.605,
}

BA_DIAL_CAREER = {
    "GPT-5.1":       0.539,
    "Qwen2.5-7B":    0.593,
    "Llama-3.1-8B":  0.486,
    "Qwen2.5-72B":   0.603,
    "DeepSeek-V3":   0.532,
    "QwQ-32B":       0.575,
    "Llama-3.1-70B": 0.527,
}

HOMOG = {
    "GPT-5.1":       0.808,
    "Qwen2.5-7B":    0.694,
    "Llama-3.1-8B":  0.280,
    "Qwen2.5-72B":   0.832,
    "DeepSeek-V3":   0.798,
    "QwQ-32B":       0.669,
    "Llama-3.1-70B": 0.553,
}

MEAN_Z = {
    "GPT-5.1":       5.43,
    "Qwen2.5-7B":    1.89,
    "Llama-3.1-8B":  1.30,
    "Qwen2.5-72B":   5.35,
    "DeepSeek-V3":   4.78,
    "QwQ-32B":       8.09,
    "Llama-3.1-70B": 4.67,
}

SIG6 = {
    "GPT-5.1":       6,
    "Qwen2.5-7B":    3,
    "Llama-3.1-8B":  1,
    "Qwen2.5-72B":   6,
    "DeepSeek-V3":   6,
    "QwQ-32B":       6,
    "Llama-3.1-70B": 6,
}

# Pattern colors
PATTERN_COLOR = {
    "GPT-5.1":       "#d62728",   # self-sufficient stereotyper — red
    "Qwen2.5-72B":   "#d62728",
    "DeepSeek-V3":   "#d62728",
    "Llama-3.1-70B": "#1f77b4",   # demographic-dependent — blue
    "QwQ-32B":       "#9467bd",   # QwQ — purple (reasoning model)
    "Qwen2.5-7B":    "#2ca02c",   # default-sufficient — green
    "Llama-3.1-8B":  "#7f7f7f",   # stochastic — grey
}

PATTERN_LABEL = {
    "GPT-5.1":       "Self-sufficient stereotyper",
    "Qwen2.5-72B":   "Self-sufficient stereotyper",
    "DeepSeek-V3":   "Self-sufficient stereotyper",
    "Llama-3.1-70B": "Demographic-dependent",
    "QwQ-32B":       "Reasoning model",
    "Qwen2.5-7B":    "Default-sufficient",
    "Llama-3.1-8B":  "Stochastic",
}

SHORT = {
    "GPT-5.1":       "GPT-5.1",
    "Qwen2.5-7B":    "Qwen-7B",
    "Llama-3.1-8B":  "Llama-8B",
    "Qwen2.5-72B":   "Qwen-72B",
    "DeepSeek-V3":   "DeepSeek",
    "QwQ-32B":       "QwQ-32B",
    "Llama-3.1-70B": "Llama-70B",
}


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Evaluation Framework (JASIST-style, tight single-column)
# ════════════════════════════════════════════════════════════════════════════
def make_figure1():
    # JASIST-style palette
    C_NONE_S = "#7a7a7a"; C_NONE_B = "#efefef"
    C_USER_S = "#548235"; C_USER_B = "#d9ead3"
    C_DIAL_S = "#4472c4"; C_DIAL_B = "#c9daf8"
    C_PER  = "#2d5986"
    C_BOT  = "#b74b4b"
    C_EVBG = "#fef9ef"; C_EVBD = "#c8a830"
    C_PROF = "#fff3cd"; C_PRBD = "#ffc107"
    WHITE  = "#ffffff";  GARR  = "#999999"

    # Single-column: 3.4"x4.8", data coords 0-10 x 0-13
    fig, ax = plt.subplots(figsize=(3.4, 4.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 13)
    ax.set_facecolor(WHITE); fig.patch.set_facecolor(WHITE)
    ax.axis("off")

    RP, RR = 0.22, 0.20  # icon scale params

    def sidebar(y0, y1, clr, lbl):
        ax.add_patch(FancyBboxPatch((.05, y0), .52, y1 - y0,
                                    boxstyle="square,pad=0",
                                    fc=clr, ec="none", zorder=3))
        ax.text(.31, (y0 + y1) * .5, lbl, ha="center", va="center",
                fontsize=4.2, fontweight="bold", color=WHITE, rotation=90, zorder=4)

    def person(cx, cy):
        ax.add_patch(plt.Circle((cx, cy + RP * .55), RP * .52,
                                 fc=C_PER, ec="none", zorder=5))
        ax.add_patch(FancyBboxPatch((cx - RP * .55, cy - RP * .65), RP * 1.1, RP * .72,
                                    boxstyle="round,pad=0.02",
                                    fc=C_PER, ec="none", zorder=5))

    def robot(cx, cy):
        ax.add_patch(FancyBboxPatch((cx - RR * .65, cy - RR * .3), RR * 1.3, RR * 1.2,
                                    boxstyle="round,pad=0.06",
                                    fc=C_BOT, ec="none", zorder=5))
        for dx in (-.25, .25):
            ax.add_patch(plt.Circle((cx + dx * RR, cy + RR * .22),
                                     RR * .14, fc=WHITE, ec="none", zorder=6))
        ax.plot([cx, cx], [cy + RR * .9, cy + RR * 1.5], color=C_BOT, lw=.7, zorder=5)
        ax.add_patch(plt.Circle((cx, cy + RR * 1.6), RR * .12,
                                 fc=C_BOT, ec="none", zorder=5))

    def bub(x0, ym, w, h, clr, txt, fs=4.8):
        ax.add_patch(FancyBboxPatch((x0, ym - h * .5), w, h,
                                    boxstyle="round,pad=0.09",
                                    fc=clr, ec="none", zorder=4))
        ax.text(x0 + .17, ym, txt, fontsize=fs, color="#2d2d2d",
                va="center", ha="left", zorder=5)

    def ev_box(x0, y0, w, h, l1, l2, c2):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.1",
                                    fc=C_EVBG, ec=C_EVBD, lw=.5, zorder=4))
        ax.text(x0 + w * .5, y0 + h * .64, l1, fontsize=4.3, color="#2d2d2d",
                ha="center", va="center", fontweight="bold", zorder=5)
        ax.text(x0 + w * .5, y0 + h * .27, l2, fontsize=3.9, color=c2,
                ha="center", va="center", multialignment="center", zorder=5)

    def dn_arr(y_from, y_to, lbl):
        xc = 4.3
        ax.annotate("", xy=(xc, y_to), xytext=(xc, y_from),
                    arrowprops=dict(arrowstyle="-|>", color=GARR,
                                   lw=.6, mutation_scale=4), zorder=3)
        ax.text(xc + .28, (y_from + y_to) * .5, lbl, fontsize=4.2,
                color=GARR, va="center", ha="left", style="italic", zorder=4)

    # Layout constants
    ICX = .95; BX = 1.47; BW = 4.90
    EX  = 6.52; EW = 2.78; BRX = 9.46
    EBH = 2.2  # uniform eval box height

    # ======================================================================
    # ROW 1 -- BA_none   (y: 10.5 -> 12.7)
    # ======================================================================
    R1B, R1T = 10.5, 12.7
    sidebar(R1B, R1T, C_NONE_S, "BA_none")

    person(ICX, 12.12)
    bub(BX, 12.12, BW, .68, C_NONE_B, "How important is personal/home life?")
    robot(ICX, 11.05)
    bub(BX, 11.05, BW, .65, C_NONE_B, "Having personal time is very important.")

    # Metric text (no eval box for baseline)
    for y, txt, fw, sty, clr in [
        (12.10, "VAA",        "bold",   "normal", "#555"),
        (11.55, "0.34–0.60",  "normal", "normal", "#555"),
        (11.05, "(baseline)", "normal", "italic",  "#888"),
    ]:
        ax.text(EX + EW * .5, y, txt, fontsize=4.4, color=clr,
                ha="center", va="center", fontweight=fw, style=sty, zorder=5)

    dn_arr(R1B - .05, R1B - .65, "+ user profile")

    # ======================================================================
    # ROW 2 -- BA_user   (y: 6.9 -> 9.8)
    # ======================================================================
    R2B, R2T = 6.9, 9.8
    sidebar(R2B, R2T, C_USER_S, "BA_user")

    # Profile strip
    ax.add_patch(FancyBboxPatch((BX, 9.22), BW, .48,
                                boxstyle="round,pad=0.08",
                                fc=C_PROF, ec=C_PRBD, lw=.4, zorder=4))
    ax.text(BX + .17, 9.46, "Age 25, Eastern Europe, trade-school",
            fontsize=4.3, color="#7a5600", va="center", ha="left",
            style="italic", zorder=5)

    person(ICX, 8.72)
    bub(BX, 8.72, BW, .65, C_USER_B, "How important is personal/home life?")
    robot(ICX, 7.65)
    bub(BX, 7.65, BW, .68, C_USER_B, "For your background, balance matters most.")

    R2_EBY0 = (R2B + R2T) * .5 - EBH * .5  # center eval box in row
    ev_box(EX, R2_EBY0, EW, EBH,
           "VAA ↑  Homog. ↑", "Stereotype-driven", "#ab6400")

    dn_arr(R2B - .05, R2B - .72, "+ chat history")

    # ======================================================================
    # ROW 3 -- BA_dialogue  (y: 0.5 -> 6.1)
    # ======================================================================
    R3B, R3T = 0.5, 6.1
    sidebar(R3B, R3T, C_DIAL_S, "BA_dialogue")

    # Prior exchange (grey -- context)
    person(ICX, 5.75)
    bub(BX, 5.75, BW, .62, "#e0e0e0", "I have a CS degree. Pursue a Master's?", fs=4.6)
    robot(ICX, 4.82)
    bub(BX, 4.82, BW, .62, "#e0e0e0", "Depends on your goals -- could help.", fs=4.6)

    # Target exchange (blue)
    person(ICX, 3.80)
    bub(BX, 3.80, BW, .62, C_DIAL_B, "How important is personal/home life?")
    robot(ICX, 2.75)
    bub(BX, 2.75, BW, .65, C_DIAL_B, "Very important -- even early in your career.")

    R3_EBY0 = (R3B + R3T) * .5 - EBH * .5  # center eval box in row
    ev_box(EX, R3_EBY0, EW, EBH,
           "Homog. ↓", "Individual signal\nemerges", "#16a34a")

    # ======================================================================
    # Consistency bracket (right side, linking eval boxes of rows 2 & 3)
    # ======================================================================
    y_bk_top = R2_EBY0 + EBH - .1
    y_bk_bot = R3_EBY0 + .1
    ax.plot([BRX, BRX], [y_bk_bot, y_bk_top], color="#666", lw=.9, zorder=4)
    for yy in (y_bk_bot, y_bk_top):
        ax.plot([BRX - .1, BRX + .1], [yy, yy], color="#666", lw=.9, zorder=4)
    ax.text(BRX, (y_bk_top + y_bk_bot) * .5, "Consis-\ntency",
            ha="center", va="center", fontsize=3.8, color="#555",
            fontweight="bold", rotation=90, zorder=5)

    fig.savefig(OUT / "Figure1.pdf", dpi=300, bbox_inches="tight", facecolor=WHITE)
    print("Saved Figure1.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Alignment-Homogenization Paradox (scatterplot)
# ════════════════════════════════════════════════════════════════════════════
def make_figure2():
    fig, ax = plt.subplots(figsize=(5.2, 4.2))

    human_baseline = 0.51

    # Distinct marker shape per model so the overlapping top cluster stays readable
    MARKERS = {
        "GPT-5.1":       "^",   # triangle-up
        "Qwen2.5-72B":   "s",   # square
        "DeepSeek-V3":   "o",   # circle
        "QwQ-32B":       "D",   # diamond
        "Llama-3.1-70B": "v",   # triangle-down
        "Qwen2.5-7B":    "P",   # thick-plus
        "Llama-3.1-8B":  "X",   # thick-cross
    }

    for m in MODELS_ORDERED:
        x = BA_USER[m]
        y = HOMOG[m]
        c = PATTERN_COLOR[m]
        ax.scatter(x, y, color=c, marker=MARKERS[m], s=110,
                   zorder=5, edgecolors="white", linewidth=0.8)

    # Label positions: top cluster pulled into a stacked right column with leader lines
    label_cfg = {
        # (text_x, text_y, ha, va)
        "Llama-3.1-8B":  (0.500, 0.248, "left",  "top"),
        "Qwen2.5-7B":    (0.524, 0.716, "right", "center"),
        "Llama-3.1-70B": (0.591, 0.522, "right", "center"),
        "QwQ-32B":       (0.591, 0.682, "right", "center"),
        # Top cluster — stacked column to the right, 0.040 spacing (1.5× line height)
        "DeepSeek-V3":   (0.655, 0.775, "left", "center"),
        "GPT-5.1":       (0.655, 0.815, "left", "center"),
        "Qwen2.5-72B":   (0.655, 0.855, "left", "center"),
    }

    for m in MODELS_ORDERED:
        x = BA_USER[m]
        y = HOMOG[m]
        c = PATTERN_COLOR[m]
        tx, ty, ha, va = label_cfg[m]
        dist = ((tx - x) ** 2 + (ty - y) ** 2) ** 0.5
        ap = dict(arrowstyle="-", color="#aaaaaa", lw=0.9,
                  shrinkA=3, shrinkB=2) if dist > 0.012 else None
        ax.annotate(SHORT[m], xy=(x, y), xytext=(tx, ty),
                    fontsize=8, color=c, fontweight="bold",
                    ha=ha, va=va, arrowprops=ap, clip_on=False)

    # Human baseline
    ax.axhline(human_baseline, color="#444", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.text(0.478, human_baseline + 0.014, "Human baseline (51%)",
            fontsize=7.5, color="#444", va="bottom")

    # Trend line (excluding Llama-8B outlier)
    trend_ms = [m for m in MODELS_ORDERED if m != "Llama-3.1-8B"]
    xs = np.array([BA_USER[m] for m in trend_ms])
    ys = np.array([HOMOG[m] for m in trend_ms])
    z  = np.polyfit(xs, ys, 1)
    xfit = np.linspace(0.540, 0.628, 60)
    ax.plot(xfit, np.polyval(z, xfit), color="#bbbbbb", linestyle=":", linewidth=1.2)

    # Legend — color + shape pairs for full disambiguation
    legend_items = [
        plt.scatter([], [], color="#d62728", marker="o", s=80, label="Self-sufficient stereotyper"),
        plt.scatter([], [], color="#1f77b4", marker="v", s=80, label="Demographic-dependent"),
        plt.scatter([], [], color="#9467bd", marker="D", s=80, label="Reasoning model (QwQ)"),
        plt.scatter([], [], color="#2ca02c", marker="P", s=80, label="Default-sufficient"),
        plt.scatter([], [], color="#7f7f7f", marker="X", s=80, label="Stochastic"),
    ]
    ax.legend(handles=legend_items, fontsize=7.5, loc="lower right",
              framealpha=0.88, edgecolor="#cccccc")

    ax.set_xlabel("Profile VAA (per-user Pearson $r$)")
    ax.set_ylabel("Homogenization rate")
    ax.set_title("Alignment-Homogenization Paradox", fontweight="bold")
    ax.set_xlim(0.470, 0.685)
    ax.set_ylim(0.18, 0.92)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "Figure2.pdf", bbox_inches="tight")
    print("Saved Figure2.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Personalization Spectrum (grouped bar chart)
# ════════════════════════════════════════════════════════════════════════════
def make_figure3():
    # Sort models by BA_none VAA ascending (weakest default first)
    sorted_models = sorted(MODELS_ORDERED, key=lambda m: BA_NONE[m])

    labels   = [SHORT[m] for m in sorted_models]
    none_v   = [BA_NONE[m] for m in sorted_models]
    user_v   = [BA_USER[m] for m in sorted_models]
    dial_v   = [BA_DIAL_CAREER[m] for m in sorted_models]

    n = len(sorted_models)
    x = np.arange(n)
    w = 0.25

    fig, ax = plt.subplots(figsize=(6.5, 3.8))

    bars_none = ax.bar(x - w,     none_v, w, label="BA_none",       color="#bdbdbd", edgecolor="white", linewidth=0.5)
    bars_user = ax.bar(x,         user_v, w, label="BA_user",       color="#1f77b4", edgecolor="white", linewidth=0.5)
    bars_dial = ax.bar(x + w,     dial_v, w, label="BA_dial career",color="#ff7f0e", edgecolor="white", linewidth=0.5)

    # Annotate delta (BA_user - BA_none) above the user bar
    for i, m in enumerate(sorted_models):
        delta = BA_USER[m] - BA_NONE[m]
        sign  = "+" if delta >= 0 else ""
        col   = "#1a6b1a" if delta > 0 else "#a00"
        ax.text(x[i], user_v[i] + 0.006, f"{sign}{delta:.2f}",
                ha="center", va="bottom", fontsize=7, color=col, fontweight="bold")

    # Shade background by pattern
    PATTERN_BG = {
        "Llama-70B":  "#EBF3FB",  # demographic-dependent
        "QwQ-32B":    "#F3EBF9",  # reasoning
        "Llama-8B":   "#F5F5F5",  # stochastic
        "DeepSeek":   "#FEF3F3",  # self-sufficient
        "Qwen-72B":   "#FEF3F3",
        "GPT-5.1":    "#FEF3F3",
        "Qwen-7B":    "#F0FAF0",  # default-sufficient
    }
    for i, lbl in enumerate(labels):
        bg = PATTERN_BG.get(lbl, "white")
        ax.axvspan(x[i] - 0.45, x[i] + 0.45, color=bg, alpha=0.45, zorder=0)

    # Two-line x-tick labels: model name + pattern category
    pattern_inline = {
        "Llama-70B":  "Demo-dep.",
        "QwQ-32B":    "Reasoning",
        "Llama-8B":   "Stochastic",
        "DeepSeek":   "Self-suff.",
        "Qwen-72B":   "Self-suff.",
        "GPT-5.1":    "Self-suff.",
        "Qwen-7B":    "Default-suff.",
    }
    tick_labels = [f"{lbl}\n{pattern_inline.get(lbl, '')}" for lbl in labels]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8.5)
    ax.set_ylabel("Per-user VAA (Pearson $r$)")
    ax.set_title("Personalization Spectrum: Default vs. Profile vs. Dialogue",
                 fontweight="bold")
    ax.set_ylim(0.30, 0.68)
    # Legend below the plot (horizontal) to avoid overlapping tall bars
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3,
              framealpha=0.9, edgecolor="#ccc", fontsize=8.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(OUT / "Figure3.pdf", bbox_inches="tight")
    print("Saved Figure3.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — Scaling Effect (two-panel)
# ════════════════════════════════════════════════════════════════════════════
def make_figure4():
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.4))

    families = {
        "Llama": {
            "small": "Llama-3.1-8B",
            "large": "Llama-3.1-70B",
            "small_label": "8B",
            "large_label": "70B",
            "color":  "#1f77b4",
        },
        "Qwen": {
            "small": "Qwen2.5-7B",
            "large": "Qwen2.5-72B",
            "small_label": "7B",
            "large_label": "72B",
            "color":  "#ff7f0e",
        },
    }

    # ── left panel: homogenization rate ──────────────────────────────────
    ax = axes[0]
    for fam, info in families.items():
        xs   = [0, 1]
        ys   = [HOMOG[info["small"]], HOMOG[info["large"]]]
        sig  = [SIG6[info["small"]],  SIG6[info["large"]]]
        col  = info["color"]
        ax.plot(xs, ys, "o-", color=col, linewidth=2, markersize=8, label=fam)
        offsets = [(-0.08, "right"), (0.08, "left")]
        for (xoff, ha), xi, yi, s in zip(offsets, xs, ys, sig):
            ax.text(xi + xoff, yi, f"{yi*100:.0f}%\n({s}/6 sig)",
                    fontsize=8, color=col, va="center", ha=ha)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Smaller\n(7B / 8B)", "Larger\n(70B / 72B)"], fontsize=9)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("Homogenization rate")
    ax.set_ylim(0.15, 0.95)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1.0))
    ax.set_title("Homogenization ↑ with scale", fontweight="bold", fontsize=10)
    ax.legend(loc="lower right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── right panel: None VAA ─────────────────────────────────────────────
    ax = axes[1]
    for fam, info in families.items():
        xs  = [0, 1]
        ys  = [BA_NONE[info["small"]], BA_NONE[info["large"]]]
        col = info["color"]
        ax.plot(xs, ys, "o-", color=col, linewidth=2, markersize=8, label=fam)
        offsets = [(-0.08, "right"), (0.08, "left")]
        for (xoff, ha), xi, yi in zip(offsets, xs, ys):
            ax.text(xi + xoff, yi, f"{yi:.3f}",
                    fontsize=8, color=col, va="center", ha=ha)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Smaller\n(7B / 8B)", "Larger\n(70B / 72B)"], fontsize=9)
    ax.set_xlim(-0.45, 1.45)
    ax.set_ylabel("None VAA (per-user Pearson $r$)")
    ax.set_ylim(0.28, 0.64)
    ax.set_title("Default culture ↓ with scale", fontweight="bold", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle("Within-Family Scaling: More Stereotyping, Weaker Defaults",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Figure4.pdf")
    print("Saved Figure4.pdf")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — Radar charts: WVS category profiles by demographic group
#             Two panels: Age (young vs old) and SES (low vs high)
#             Lines: Human ground truth + GPT-5.1 + Llama-3.1-8B
# ════════════════════════════════════════════════════════════════════════════

CAT_SHORT = {
    "Social Values, Norms, Stereotypes":                          "Social\nValues",
    "Social Capital, Trust and Organizational Membership":         "Social\nCapital",
    "Economic Values":                                             "Economic",
    "Perceptions of Corruption":                                   "Corruption",
    "Perceptions of Migration":                                    "Migration",
    "Perceptions of Security":                                     "Security",
    "Perceptions about Science and Technology":                    "Science",
    "Religious Values":                                            "Religious",
    "Ethical Values":                                              "Ethical",
    "Political Interest and Political Participation":              "Political\nActivity",
    "Political Culture and Political Regimes":                     "Political\nCulture",
}

def _build_q_cat_map():
    """Return {question_id: category_name} from the gpt-5.1 JSONL."""
    q_to_cat = {}
    with open(ROOT / "wvs_values_results/gpt-5.1/BA_user_values_results/total_1000.jsonl") as f:
        entry = json.loads(f.readline())
    uid = list(entry.keys())[0]
    for cat, qs in entry[uid].items():
        for qd in qs:
            for qid in qd:
                q_to_cat[qid] = cat
    return q_to_cat


def _load_model_profiles(jsonl_path, q_to_cat, categories):
    """Return {user_id: {category: mean_option_id}} from a BA_user JSONL."""
    profiles = {}
    with open(jsonl_path) as f:
        for line in f:
            entry = json.loads(line)
            for uid, udata in entry.items():
                cat_vals = {c: [] for c in categories}
                for cat, qs in udata.items():
                    if cat not in cat_vals:
                        continue
                    for qd in qs:
                        for qid, resp in qd.items():
                            if isinstance(resp, dict) and "option_id" in resp:
                                cat_vals[cat].append(resp["option_id"])
                profiles[uid] = {
                    c: (sum(v) / len(v) if v else None)
                    for c, v in cat_vals.items()
                }
    return profiles


def _group_means(profiles, user_ids, categories, q_norms):
    """Per-category mean for a user subset, normalized to [0,1] using q_norms."""
    cat_sums = {c: 0.0 for c in categories}
    cat_counts = {c: 0 for c in categories}
    for uid in user_ids:
        if uid not in profiles:
            continue
        for cat, val in profiles[uid].items():
            if val is None:
                continue
            lo, hi = q_norms.get(cat, (1, 10))
            norm = (val - lo) / max(hi - lo, 1)
            cat_sums[cat] += norm
            cat_counts[cat] += 1
    return {c: (cat_sums[c] / cat_counts[c] if cat_counts[c] > 0 else 0.5)
            for c in categories}


def _human_group_means(merged_df, mask, categories, q_to_cat, q_norms):
    """Per-category mean for a human subset, normalized to [0,1]."""
    sub = merged_df[mask]
    cat_sums = {c: 0.0 for c in categories}
    cat_counts = {c: 0 for c in categories}
    for qid, cat in q_to_cat.items():
        if qid not in sub.columns or cat not in categories:
            continue
        lo, hi = q_norms.get(cat, (1, 10))
        vals = sub[qid].dropna()
        norm_vals = (vals - lo) / max(hi - lo, 1)
        cat_sums[cat] += norm_vals.sum()
        cat_counts[cat] += len(norm_vals)
    return {c: (cat_sums[c] / cat_counts[c] if cat_counts[c] > 0 else 0.5)
            for c in categories}


def make_figure6():
    """Gradient line plot: pairwise L2 distance vs. age span for 10 age-group pairs.
    Directly visualises the Spearman rho=0.64 claim in §Analysis."""
    q_to_cat = _build_q_cat_map()
    categories = list(CAT_SHORT.keys())

    # ── normalization bounds ─────────────────────────────────────────────────
    vals_df  = pd.read_csv(ROOT / "datasets/wvs_benchmarks/sampled_values_df.csv")
    demo_df  = pd.read_csv(ROOT / "datasets/wvs_benchmarks/sampled_demographic_features.csv")
    demo_df["D_INTERVIEW"] = demo_df["D_INTERVIEW"].astype(str)
    vals_df["D_INTERVIEW"] = vals_df["D_INTERVIEW"].astype(str)
    merged = demo_df.merge(vals_df, on="D_INTERVIEW")

    q_norms = {}
    for qid, cat in q_to_cat.items():
        if qid not in merged.columns:
            continue
        col = merged[qid].dropna()
        lo, hi = col.min(), col.max()
        if cat not in q_norms:
            q_norms[cat] = (lo, hi)
        else:
            q_norms[cat] = (min(q_norms[cat][0], lo), max(q_norms[cat][1], hi))

    # ── 5 age bins with representative midpoints ─────────────────────────────
    age_bins = [
        ("<30",   merged["age"] < 30,                                        22),
        ("30-40", (merged["age"] >= 30) & (merged["age"] < 40),             35),
        ("40-50", (merged["age"] >= 40) & (merged["age"] < 50),             45),
        ("50-60", (merged["age"] >= 50) & (merged["age"] < 60),             55),
        (">60",   merged["age"] >= 60,                                       67),
    ]

    # ── load model profiles ─────────────────────────────────────────────────
    gpt_profs   = _load_model_profiles(
        ROOT / "wvs_values_results/gpt-5.1/BA_user_values_results/total_1000.jsonl",
        q_to_cat, categories)
    llama_profs = _load_model_profiles(
        ROOT / "wvs_values_results/Llama-3.1-8B-Instruct/BA_user_values_results/total_1000.jsonl",
        q_to_cat, categories)

    # ── per-bin category-mean vectors (11-dim, normalised) ───────────────────
    def bin_vec(means_dict):
        return np.array([means_dict[c] for c in categories])

    human_vecs, gpt_vecs, llama_vecs = {}, {}, {}
    for label, mask, _ in age_bins:
        ids = set(merged[mask]["D_INTERVIEW"])
        human_vecs[label] = bin_vec(_human_group_means(merged, mask, categories, q_to_cat, q_norms))
        gpt_vecs[label]   = bin_vec(_group_means(gpt_profs,   ids, categories, q_norms))
        llama_vecs[label] = bin_vec(_group_means(llama_profs, ids, categories, q_norms))

    # ── all 10 pairwise combinations, sorted by age span ────────────────────
    bin_labels = [b[0] for b in age_bins]
    bin_mids   = {b[0]: b[2] for b in age_bins}

    rows = []
    for i, a in enumerate(bin_labels):
        for b in bin_labels[i + 1:]:
            span = abs(bin_mids[a] - bin_mids[b])
            rows.append({
                "label": f"{a} vs {b}",
                "span":  span,
                "human": float(np.linalg.norm(human_vecs[a] - human_vecs[b])),
                "gpt":   float(np.linalg.norm(gpt_vecs[a]   - gpt_vecs[b])),
                "llama": float(np.linalg.norm(llama_vecs[a]  - llama_vecs[b])),
            })
    rows.sort(key=lambda r: (r["span"], r["label"]))

    x        = list(range(len(rows)))
    xlabels  = [r["label"]  for r in rows]
    h_dists  = [r["human"]  for r in rows]
    g_dists  = [r["gpt"]    for r in rows]
    l_dists  = [r["llama"]  for r in rows]

    spans_arr = np.array([r["span"] for r in rows])
    h_dists   = np.array(h_dists)
    g_dists   = np.array(g_dists)
    l_dists   = np.array(l_dists)

    # ── jitter x=10 collision (two pairs) by ±1.5 yr ────────────────────────
    x_h = spans_arr.copy().astype(float)
    x_g = spans_arr.copy().astype(float)
    x_l = spans_arr.copy().astype(float)
    tie_idxs = [i for i, s in enumerate(spans_arr) if s == 10]
    if len(tie_idxs) == 2:
        x_h[tie_idxs[0]] -= 1.5;  x_h[tie_idxs[1]] += 1.5
        x_g[tie_idxs[0]] -= 1.5;  x_g[tie_idxs[1]] += 1.5
        x_l[tie_idxs[0]] -= 1.5;  x_l[tie_idxs[1]] += 1.5

    # ── OLS trend lines ──────────────────────────────────────────────────────
    xs_fit = np.linspace(spans_arr.min() - 2, spans_arr.max() + 2, 120)
    coef_g = np.polyfit(spans_arr, g_dists, 1)
    coef_l = np.polyfit(spans_arr, l_dists, 1)

    # ── scatter plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 3.6))

    # Human baseline — small grey circles, no trend line
    ax.scatter(x_h, h_dists, color="#888888", marker="o", s=28, zorder=3,
               alpha=0.75, label="Human ground truth")

    # Llama-3.1-8B — blue triangles + flat OLS
    ax.scatter(x_l, l_dists, color="#1f77b4", marker="^", s=48, zorder=4,
               label="Llama-3.1-8B (Homog Rate 28.0%)")
    ax.plot(xs_fit, np.polyval(coef_l, xs_fit),
            color="#1f77b4", lw=1.3, linestyle="-", alpha=0.45, zorder=2)

    # GPT-5.1 — red squares + positive-slope OLS (primary claim)
    ax.scatter(x_g, g_dists, color="#d62728", marker="s", s=52, zorder=5,
               label=r"GPT-5.1 (Homog Rate 80.8%)")
    ax.plot(xs_fit, np.polyval(coef_g, xs_fit),
            color="#d62728", lw=1.5, linestyle="--", alpha=0.65, zorder=2)

    # Annotate ρ on the GPT-5.1 trend line
    mid_x = xs_fit[len(xs_fit) // 2] - 4
    mid_y = np.polyval(coef_g, mid_x) + 0.008
    ax.text(mid_x, mid_y, r"$\rho=0.64$", color="#d62728", fontsize=7.5,
            fontstyle="italic")

    # Annotate extreme pair — label directly above the point
    idx_max = int(np.argmax(x_g))
    ax.annotate("<30 vs >60",
                xy=(x_g[idx_max], g_dists[idx_max]),
                xytext=(x_g[idx_max] - 2, g_dists[idx_max] + 0.018),
                fontsize=6.5, color="#d62728", ha="right",
                arrowprops=dict(arrowstyle="-", color="#d62728", lw=0.7))

    ax.set_xlabel("Age span between groups (years)", fontsize=8.5)
    ax.set_ylabel("Pairwise response distance (L2)", fontsize=8.5)
    ax.set_title("Stereotype magnitude scales with age span (GPT-5.1 vs Llama-3.1-8B)",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, edgecolor="#cccccc")
    ax.grid(alpha=0.18, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(OUT / "Figure6.pdf", bbox_inches="tight")
    print("Saved Figure6.pdf")
    plt.close(fig)


if __name__ == "__main__":
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure6()
    print(f"\nAll figures saved to {OUT}")
