"""Step 5. Robustness of the joint term (Table 5), plus a sparse model that is not in the manuscript.

Each row re-estimates Table 2, Model 3 with one change (sample, estimator, outcome, measure of
the moderator or of facial energy, or added controls) and reports the interaction, the slope of
facial energy at the 90th and 10th percentile of the moderator, and the Johnson-Neyman share.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from common import Y, Y2, controls, fe_fit, fit, johnson_neyman, lincom, load, z


def add_measures(df):
    """Alternative measures used in Table 5."""
    df = df.copy()
    wins = lambda s: z(s.astype(float).clip(*s.astype(float).quantile([.01, .99])))
    df["neg_vader"], df["neg_lm"], df["neg_nrc"] = z(df.w_vader_neg), z(df.w_lm_neg), z(df.w_nrc_negative)
    df["low_enthusiasm"] = z(-df.verbal_enthusiasm_axis)     # word-vector axis, reversed
    df["face_net_mouth"] = z(df.face_artic_free)              # facial arousal net of mouth opening
    df["numbers"] = wins(df.numbers_per100)                   # content controls
    df["traction"] = z(df.traction_score_0to5)
    df["mission"] = z(df.mission_vs_commercial_axis)
    df["ask_stated"] = df.ask_usd.notna().astype(float)
    df["revenue_figure"] = df.any_revenue_figure.astype(float)
    df["formal_style"] = z(df.Analytic)                       # LIWC-22 Analytic
    df["face_x_formal"] = df.face_energy * df.formal_style
    return df


def row(df, label, mod="neg_words", f="face_energy", extra=(), sample=None, y=Y, fe=False, C=None):
    d = (df if sample is None else df[sample]).copy()
    d["_M"], d["_F"] = d[mod], d[f]
    d["_X"] = d._F * d._M
    if C is None:
        C = controls(d, seasons=not fe)
    X = C + ["_F", "voice_energy", "_M", "_X"] + list(extra)
    if fe:
        params, bse, cov, d2 = fe_fit(d, X, y=y)
        pv = pd.Series(2 * norm.sf(np.abs(params / bse)), index=params.index)
    else:
        r, d2 = fit(d, X, y=y)
        params, bse, pv, cov = r.params, r.bse, r.pvalues, r.cov_params()
    s90 = lincom(params, cov, {"_F": 1, "_X": d2._M.quantile(.9)})
    s10 = lincom(params, cov, {"_F": 1, "_X": d2._M.quantile(.1)})
    share, _, nshare = johnson_neyman(params, cov, d2._M.values, f="_F", fx="_X")
    return dict(label=label, b=float(params["_X"]), se=float(bse["_X"]), p=float(pv["_X"]),
                slope90=s90, slope10=s10, jn_share=share, jn_negative_share=nshare,
                n=len(d2), episodes=int(d2.vid_id.nunique()))


def run(df=None, quick=False):
    df = add_measures(load() if df is None else df)
    content = ["numbers", "traction", "mission", "ask_stated", "revenue_figure"]
    rows = [
        row(df, "Full sample (Table 2, Model 3)"),
        row(df, "Random half A", sample=df.random_half == "A"),
        row(df, "Random half B", sample=df.random_half == "B"),
        row(df, "Episode fixed effects", fe=True),
        row(df, "Second outcome: pitch success", y=Y2),
        row(df, "Negative words: VADER", mod="neg_vader"),
        row(df, "Negative words: NRC", mod="neg_nrc"),
        row(df, "Negative words: average of the three dictionaries", mod="neg_comp"),
        row(df, "Word vectors: low verbal enthusiasm", mod="low_enthusiasm"),
        row(df, "Facial arousal net of mouth opening", f="face_net_mouth"),
        row(df, "Adding content controls", extra=content),
        row(df, "Adding formal style x facial arousal", extra=["formal_style", "face_x_formal"]),
        row(df, "Single active speaker only", sample=df.n_active_speakers == 1),
        row(df, "Without the two unclassified pitches", sample=df.industry_cat.notna()),
    ]
    R = {"table5": rows}
    C = controls(df)
    # Negative words and low verbal enthusiasm in the same model: do they capture one moderator?
    d = df.assign(face_x_lowenth=df.face_energy * df.low_enthusiasm)
    r = fit(d, C + ["face_energy", "voice_energy", "neg_words", "face_x_neg", "low_enthusiasm", "face_x_lowenth"])[0]
    R["neg_and_low_enthusiasm"] = {c: [float(r.params[c]), float(r.pvalues[c])] for c in ["face_x_neg", "face_x_lowenth"]}
    r = fit(df, C + ["face_energy", "voice_energy", "neg_words", "face_x_neg", "formal_style", "face_x_formal"])[0]
    R["neg_and_formal_style"] = {c: [float(r.params[c]), float(r.pvalues[c])] for c in ["face_x_neg", "face_x_formal"]}
    R["r_neg_low_enthusiasm"] = float(df.neg_words.corr(df.low_enthusiasm))
    R["r_face_net_mouth"] = float(df.face_energy.corr(df.face_net_mouth))
    R["r_neg_components"] = df[["neg_vader", "neg_lm", "neg_nrc"]].corr().values.tolist()

    # ---- Sparse model, not reported in the manuscript
    # 156 funded pitches against 26 parameters in Model 3 (about 6 events per parameter).
    # Sparse model: drop the industry dummies and season fixed effects (12 parameters).
    sparse = controls(df, industry=False, seasons=False)
    R["extra_sparse"] = row(df, "Sparse controls (no industry, no season)", C=sparse)
    R["extra_sparse"]["n_params"] = len(sparse) + 5
    return R


if __name__ == "__main__":
    from common import fb, pct
    R = run()
    for x in R["table5"] + [R["extra_sparse"]]:
        print(f"{x['label']:45s} {fb(x['b'], x['se'], x['p']):28s} slope90 {fb(*x['slope90']):26s} slope10 {fb(*x['slope10']):26s}"
              f" JN {pct(x['jn_share'])}% (neg {pct(x['jn_negative_share'])}%)  n = {x['n']} ({x['episodes']})")
    print("NEG and low enthusiasm together:", R["neg_and_low_enthusiasm"])
    print("NEG and formal style together:", R["neg_and_formal_style"])
