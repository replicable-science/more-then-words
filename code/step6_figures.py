"""Step 6. Figures 3, 4, and 5 (one per joint test), same size, no titles inside.

1  Contradictions are reconciled: raw funding rates by thirds of facial energy and negative words.
2  Unclear scope is bounded: funding gain from a passionate face along negative words.
3  Redundancy is pruned: correlations behind the jingle and the jangle.
"""
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from common import FIG, Y, controls, fit, johnson_neyman, load, thirds

SURF, INK, INK2, GRID, B1, B2 = "#ffffff", "#1F2A37", "#4b5563", "#e4e3df", "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 15, "axes.edgecolor": "#9ca3af", "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.labelsize": 16, "xtick.labelsize": 15, "ytick.labelsize": 15})
SIZE, DPI = (10, 6.2), 200


def _axes():
    fig = plt.figure(figsize=SIZE, facecolor=SURF)
    a = fig.add_axes([0.13, 0.14, 0.83, 0.80])
    a.set_facecolor(SURF)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.grid(axis="y", color=GRID, lw=1)
    a.set_axisbelow(True)
    return fig, a


def graph1(df, path):
    t_face = thirds(df.face_energy, ["calm", "mid", "pass"])
    t_neg = thirds(df.neg_words, ["few", "mid", "many"])
    T = df.groupby([t_neg, t_face])[Y].mean().unstack()
    fig, a = _axes()
    x = np.array([0, 1])
    a.set_xlim(-0.1, 1.1); a.set_ylim(0.10, 0.50)
    lines = {"few": (B1, "Upbeat pitch (few negative words)", +1), "many": (B2, "Problem-focused pitch (many negative words)", -1)}
    for row, (col, lab, side) in lines.items():
        y = np.array([T.loc[row, "calm"], T.loc[row, "pass"]])
        a.plot(x, y, color=col, lw=3.2, marker="o", ms=13, mec=SURF, mew=2.5, zorder=3, solid_capstyle="round")
        other = np.array([T.loc["many" if row == "few" else "few", c] for c in ["calm", "pass"]])
        for xi, yi, oi in zip(x, y, other):
            up = yi >= oi
            a.text(xi, yi + (0.016 if up else -0.016), f"{yi:.0%}", ha="center", va="bottom" if up else "top",
                   color=INK, fontsize=16, fontweight="bold")
        xl = 0.36
        yl = y[0] + (y[1] - y[0]) * xl
        p0, p1 = a.transData.transform((0, y[0])), a.transData.transform((1, y[1]))
        ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        a.text(xl, yl + side * 0.014, lab, color=col, fontsize=15, ha="center", va="bottom" if side > 0 else "top",
               rotation=ang, rotation_mode="anchor", fontweight="bold")
    a.set_xticks(x); a.set_xticklabels(["Calm face", "Passionate face"], fontsize=16)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    a.set_ylabel("Share of pitches funded")
    fig.savefig(path, dpi=DPI, facecolor=SURF); plt.close(fig)


def graph2(df, curve, path):
    pcts, pt, lo, hi = (np.array(curve[k]) for k in ["pct", "gain", "lo", "hi"])
    r3, d3 = fit(df, controls(df) + ["face_energy", "voice_energy", "neg_words", "face_x_neg"])
    thr = johnson_neyman(r3.params, r3.cov_params(), d3.neg_words.values)[1]
    thr_pct = (d3.neg_words < thr).mean() * 100
    fig, a = _axes()
    a.axvspan(thr_pct, 100, color=B2, alpha=0.09, lw=0, zorder=0)
    a.fill_between(pcts, 100 * lo, 100 * hi, color=B1, alpha=0.16, lw=0, zorder=2)
    a.plot(pcts, 100 * pt, color=B1, lw=3.2, zorder=3)
    a.axhline(0, color=INK2, lw=1.3, zorder=2)
    a.axvline(thr_pct, color=INK2, lw=1.3, ls=(0, (4, 3)), zorder=2)
    a.set_xlim(0, 100); a.set_ylim(-30, 58)
    a.set_yticks([-20, -10, 0, 10, 20, 30, 40, 50]); a.set_yticklabels(["-20", "-10", "0", "+10", "+20", "+30", "+40", "+50"])
    a.set_xticks([])
    a.set_ylabel("Funding gain from a passionate face\n(percentage points)")
    a.text(0, -0.035, "Upbeat pitch", transform=a.transAxes, ha="left", va="top", fontsize=16, color=INK)
    a.text(1, -0.035, "Problem-focused pitch", transform=a.transAxes, ha="right", va="top", fontsize=16, color=INK)
    a.annotate("", xy=(0.74, -0.06), xytext=(0.20, -0.06), xycoords="axes fraction",
               arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, mutation_scale=16))
    a.text(thr_pct + 1.8, 56, f"Passion pays significantly\nin the {100 - thr_pct:.0f}% most\nproblem-focused pitches",
           color=INK, fontsize=15, va="top", ha="left")
    a.text(2, 56, "No evidence that passion\npays (or costs) here", color=INK2, fontsize=15, va="top", ha="left")
    fig.savefig(path, dpi=DPI, facecolor=SURF); plt.close(fig)


def graph3(df, path):
    order = ["face_energy", "voice_energy", "joy_words", "joy", "smile", "pos_face"]
    labels = ["Facial energy", "Vocal energy", "Joyful words", "Joy", "Smile", "Positive face"]
    R = df[order].corr().values
    fig = plt.figure(figsize=SIZE, facecolor=SURF)
    a = fig.add_axes([0.17, 0.14, 0.50, 0.80])
    cmap = LinearSegmentedColormap.from_list("seqblue", ["#f3f6fb", "#c9daf3", "#7fa9e6", "#2a78d6", "#18457f"])
    a.imshow(np.clip(R, 0, 1), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(6):
        for j in range(6):
            v = R[i, j]
            s = "1" if i == j else f"{v:.2f}".replace("0.", ".").replace("-.00", ".00")
            a.text(j, i, s, ha="center", va="center", fontsize=16, color="white" if v > 0.55 else INK)
    a.set_xticks(range(6)); a.set_yticks(range(6))
    a.set_xticklabels(["Facial\nenergy", "Vocal\nenergy", "Joyful\nwords", "Joy", "Smile", "Positive\nface"], fontsize=14)
    a.set_yticklabels(labels, fontsize=15)
    a.tick_params(length=0)
    for s in a.spines:
        a.spines[s].set_visible(False)
    for k in range(7):
        a.axhline(k - 0.5, color=SURF, lw=2.5); a.axvline(k - 0.5, color=SURF, lw=2.5)
    a.add_patch(Rectangle((-0.5, -0.5), 3, 3, fill=False, ec=B2, lw=3, ls=(0, (5, 3)), zorder=5))
    a.add_patch(Rectangle((2.5, 2.5), 3, 3, fill=False, ec=INK, lw=3, zorder=6))
    L = fig.add_axes([0.70, 0.14, 0.28, 0.80]); L.axis("off"); L.set_xlim(0, 1); L.set_ylim(0, 1)
    L.add_patch(Rectangle((0.0, 0.86), 0.09, 0.07, fill=False, ec=B2, lw=3, ls=(0, (4, 2))))
    L.text(0.14, 0.895, "Jingle", fontsize=17, fontweight="bold", color=B2, va="center")
    L.text(0.0, 0.80, "\n".join(textwrap.wrap("One label, displayed passion, for three signals from face, voice, and words "
                                               "that barely overlap (r ≤ .12)", 30)), fontsize=15, color=INK, va="top", linespacing=1.3)
    L.add_patch(Rectangle((0.0, 0.40), 0.09, 0.07, fill=False, ec=INK, lw=3))
    L.text(0.14, 0.435, "Jangle", fontsize=17, fontweight="bold", color=INK, va="center")
    L.text(0.0, 0.34, "\n".join(textwrap.wrap("Three labels, joy, smile, and positive face, for one signal (r = .65 to .81) "
                                               "that does not predict funding", 30)), fontsize=15, color=INK, va="top", linespacing=1.3)
    fig.savefig(path, dpi=DPI, facecolor=SURF); plt.close(fig)


def run(df=None, quick=False, curve=None):
    df = load() if df is None else df
    if curve is None:
        from step2_boundary import gain_curve
        r3, d3 = fit(df, controls(df) + ["face_energy", "voice_energy", "neg_words", "face_x_neg"])
        curve = gain_curve(r3, d3, draws=300 if quick else 2000)
    FIG.mkdir(parents=True, exist_ok=True)
    paths = [FIG / "graph1_reconciliation.png", FIG / "graph2_boundary.png", FIG / "graph3_pruning.png"]
    graph1(df, paths[0]); graph2(df, curve, paths[1]); graph3(df, paths[2])
    return {"figures": [str(p.relative_to(FIG.parents[1])) for p in paths]}


if __name__ == "__main__":
    import sys
    print(run(quick="--quick" in sys.argv))
