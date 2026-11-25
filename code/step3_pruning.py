"""Step 3. Redundancy -> pruning, made possible by objectivity (Table 4).

Jingle: one label (displayed passion) for signals that studies read from different channels:
facial energy, vocal energy, joyful words. Do they coincide?
Jangle: three labels (joy, smile, positive face) from separate literatures. Are they one signal,
and does that signal predict funding?
"""
import numpy as np
from scipy.stats import chi2, pearsonr

from common import Y2, controls, fit, load, summarize, coef, z

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

    # Table 4, Panel B. Dominance beyond vocal energy, and the three face scores beyond their index.
    R["dominance_r"] = float(pearsonr(df.voice_energy, df.voice_dom)[0])
    dom_full = fit(df, C + ["voice_energy", "voice_dom"])[0]
    dom_red = fit(df, C + ["voice_energy"])[0]
    dom_stat = float(2 * (dom_full.llf - dom_red.llf))
    R["dominance_lr"] = dict(chi2=dom_stat, df=1, p=float(chi2.sf(dom_stat, 1)))
    face_full = fit(df, C + JANGLE)[0]
    face_red = fit(df, C + ["pos_face_index"])[0]
    face_stat = float(2 * (face_full.llf - face_red.llf))
    R["face_beyond_index"] = dict(chi2=face_stat, df=2, p=float(chi2.sf(face_stat, 2)))

    # Table 4, Panel A. One pooled energy interaction against the three channel-specific ones.
    d = df.copy()
    d["energy_pool"] = z(d[["face_energy", "voice_energy", "verbal_arousal"]].mean(axis=1))
    d["pool_x_neg"] = d.energy_pool * d.neg_words
    base = C + ["face_energy", "voice_energy", "verbal_arousal", "neg_words"]
    apart = fit(d, base + ["face_x_neg", "voice_x_neg", "verbal_x_neg"])[0]
    pooled = fit(d, base + ["pool_x_neg"])[0]
    pool_stat = float(2 * (apart.llf - pooled.llf))
    R["pooled_interaction"] = coef(pooled, "pool_x_neg")
    R["pooled_lr"] = dict(chi2=pool_stat, df=2, p=float(chi2.sf(pool_stat, 2)))

    # Table 4, Panel B. Five verbal positivity measures against their index.
    v = df.copy()
    v["vader_pos"] = z(v.w_vader_pos)
    v["nrc_pos"] = z(v.w_nrc_positive)
    verbal = ["joy_words", "pos_words", "vader_pos", "nrc_pos", "verbal_valence"]
    v["verb_pos_index"] = z(v[verbal].mean(axis=1))
    verb_only = fit(v, C + ["verb_pos_index"])[0]
    verb_full = fit(v, C + verbal)[0]
    verb_stat = float(2 * (verb_full.llf - verb_only.llf))
    R["verbal_index"] = coef(verb_only, "verb_pos_index")
    R["verbal_beyond_index"] = dict(chi2=verb_stat, df=4, p=float(chi2.sf(verb_stat, 4)))

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
