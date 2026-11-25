"""Step 3. Redundancy -> pruning, made possible by objectivity (Table 4).

Jingle: one label (displayed passion) for signals that studies read from different channels:
facial energy, vocal energy, joyful words. Do they coincide?
Jangle: three labels (joy, smile, positive face) from separate literatures. Are they one signal,
and does that signal predict funding?
"""
import numpy as np
from scipy.stats import pearsonr

from common import Y2, controls, fit, load, summarize, coef

JINGLE = ["face_energy", "voice_energy", "joy_words"]
JANGLE = ["joy", "smile", "pos_face"]


def alpha(frame):
    """Cronbach's alpha of the columns of a data frame."""
    k = frame.shape[1]
    return float(k / (k - 1) * (1 - frame.var().sum() / frame.sum(axis=1).var()))


def run(df=None, quick=False):
    df = load() if df is None else df
    C = controls(df)
    R = {}

    # ---- Panel A: correlations
    six = JINGLE + JANGLE
    R["table4_panelA"] = dict(order=six, r=df[six].corr().values.tolist())
    R["jingle_r"] = {f"{a}|{b}": [float(x) for x in pearsonr(df[a], df[b])]
                     for a, b in [("face_energy", "voice_energy"), ("face_energy", "joy_words"), ("voice_energy", "joy_words")]}
    R["jangle_r"] = {f"{a}|{b}": float(df[a].corr(df[b])) for a, b in [("joy", "smile"), ("joy", "pos_face"), ("smile", "pos_face")]}
    R["jangle_r_halves"] = {h: {f"{a}|{b}": float(d[a].corr(d[b])) for a, b in [("joy", "smile"), ("joy", "pos_face"), ("smile", "pos_face")]}
                            for h, d in df.groupby("random_half")}
    R["alpha_pos_face_index"] = alpha(df[JANGLE])
    R["r_face_energy_with_positive_expression"] = {c: float(df.face_energy.corr(df[c])) for c in JANGLE + ["pos_face_index"]}
    R["joy_words_raw"] = dict(mean=float(df.w_nrc_joy.mean()), sd=float(df.w_nrc_joy.std()))

    # ---- Panel B: funding models
    models = {
        "M1": C + ["joy"], "M2": C + ["smile"], "M3": C + ["pos_face"],
        "M4": C + JANGLE,
        "M5": C + ["pos_face_index", "face_energy", "voice_energy"],
        "M6": C + ["pos_face_index", "face_energy", "voice_energy", "neg_words", "face_x_neg", "pfi_x_neg"],
    }
    fits = {k: fit(df, X)[0] for k, X in models.items()}
    R["table4_panelB"] = {k: summarize(r) for k, r in fits.items()}
    R["alone"] = {c: coef(fits[m], c) for m, c in [("M1", "joy"), ("M2", "smile"), ("M3", "pos_face")]}
    R["together"] = {c: coef(fits["M4"], c) for c in JANGLE}
    R["index"] = coef(fits["M5"], "pos_face_index")
    R["index_x_neg"] = coef(fits["M6"], "pfi_x_neg")
    # Second outcome: smallest p of the focal positive-expression terms across Models 1 to 6
    p2 = []
    for k, X in models.items():
        r = fit(df, X, y=Y2)[0]
        p2 += [float(r.pvalues[c]) for c in ["joy", "smile", "pos_face", "pos_face_index", "pfi_x_neg"] if c in X]
    R["second_outcome_min_p"] = min(p2)
    # Inverted U (Warnick et al., 2021): squared terms
    R["squared_p"] = {}
    for c in JANGLE + ["pos_face_index"]:
        d = df.assign(**{c + "_sq": df[c] ** 2})
        R["squared_p"][c] = float(fit(d, C + [c, c + "_sq"])[0].pvalues[c + "_sq"])
    # Joyful words: main effect and moderation of the cost of naming problems (Table 4 note)
    R["joy_words_alone"] = coef(fit(df, C + ["joy_words"])[0], "joy_words")
    R["joy_words_x_neg"] = coef(fit(df, C + ["face_energy", "voice_energy", "neg_words", "face_x_neg",
                                              "joy_words", "joywords_x_neg"])[0], "joywords_x_neg")

    # ---- Extra checks, not reported in the manuscript
    # (a) Is the jingle just unreliable measurement? Alpha across thirds of the pitch, and
    #     correlations corrected for attenuation with those alphas.
    th = {"face_energy": ["arousal_visual_norm_t1", "arousal_visual_norm_t2", "arousal_visual_norm_t3"],
          "voice_energy": ["arousal_vocal_norm_t1", "arousal_vocal_norm_t2", "arousal_vocal_norm_t3"],
          "verbal_arousal": ["arousal_text_norm_t1", "arousal_text_norm_t2", "arousal_text_norm_t3"],
          "joy_words": ["t1_nrc_joy", "t2_nrc_joy", "t3_nrc_joy"]}
    rel = {k: alpha(df[v].astype(float)) for k, v in th.items()}
    R["extra_alpha_thirds"] = rel
    R["extra_jingle_disattenuated"] = {k: v[0] / np.sqrt(rel[k.split("|")[0]] * rel[k.split("|")[1]]) for k, v in R["jingle_r"].items()}
    # (b) How large could the pruned effects be? 95% confidence intervals (log-odds per SD)
    R["extra_ci95"] = {}
    for m, c in [("M1", "joy"), ("M2", "smile"), ("M3", "pos_face"), ("M5", "pos_face_index"), ("M6", "pfi_x_neg")]:
        lo, hi = fits[m].conf_int().loc[c]
        R["extra_ci95"][c] = [float(lo), float(hi)]
    return R


if __name__ == "__main__":
    from common import fb, rhu
    R = run()
    print("Jingle r:", {k: rhu(v[0], 2) for k, v in R["jingle_r"].items()})
    print("Jangle r:", {k: rhu(v, 2) for k, v in R["jangle_r"].items()}, "alpha", rhu(R["alpha_pos_face_index"], 2))
    print("Alone:", {k: fb(*v) for k, v in R["alone"].items()})
    print("Together:", {k: fb(*v) for k, v in R["together"].items()})
    print("Index:", fb(*R["index"]), "| index x neg:", fb(*R["index_x_neg"]), "| squared p:", {k: rhu(v, 2) for k, v in R["squared_p"].items()})
    print("Second outcome min p", rhu(R["second_outcome_min_p"], 3), "| joy words", fb(*R["joy_words_alone"]), "x neg", fb(*R["joy_words_x_neg"]))
    print("Extra alpha thirds", {k: rhu(v, 2) for k, v in R["extra_alpha_thirds"].items()},
          "| disattenuated", {k: rhu(v, 2) for k, v in R["extra_jingle_disattenuated"].items()})
    print("Extra 95% CIs", {k: [rhu(x, 2) for x in v] for k, v in R["extra_ci95"].items()})
