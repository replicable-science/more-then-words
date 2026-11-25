"""Shared setup for the three joint tests in the manuscript.

Everything that more than one step needs lives here: reading the data, building the
standardized measures and the controls, and the model helpers (clustered logit,
conditional logit, linear combinations, predicted probabilities, Johnson-Neyman).

Run all steps with `python run_all.py`, or one step at a time, for example
`python code/step1_reconciliation.py`.
"""
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

warnings.filterwarnings("ignore")  # convergence chatter from bootstrap and subset fits

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
FIG = OUT / "figures"

Y = "funding_success"   # main outcome: 1 = the panel funded the venture (34%)
Y2 = "pitch_success"    # second outcome, robustness only (65%); every funded pitch has it


def z(s):
    """Standardize with the sample SD (the pandas default)."""
    s = s.astype(float)
    return (s - s.mean()) / s.std()


def load():
    """Read the 457 pitches and build every analysis variable."""
    df = pd.read_csv(DATA / "pitches.csv")
    assert len(df) == 457 and df.subsegment_id.is_unique

    # ---- Focal measures, one per channel, standardized over the 457 pitches
    df["face_energy"] = z(df.arousal_visual_norm)    # displayed passion: facial arousal, ELIM-FER
    df["voice_energy"] = z(df.arousal_vocal_norm)    # vocal arousal, wav2vec2 dimensional model
    df["voice_dom"] = z(df.dominance_vocal)          # vocal dominance, same model, mean of the five-second windows
    df["verbal_arousal"] = z(df.arousal_text_norm)   # arousal of the words, NRC-VAD lexicon
    df["verbal_valence"] = z(df.valence_text)        # valence of the words, same lexicon
    df["joy_words"] = z(df.w_nrc_joy)                # NRC joy words per 100 words (passion in text)

    # Negative words are Loughran-McDonald negative words per 100 words, standardized.
    # Positive words are the positive list of the same dictionary, standardized.
    # The three-dictionary averages stay available as neg_comp and pos_comp for Table 5.
    df["neg_words"] = z(df.w_lm_neg)
    df["pos_words"] = z(df.w_lm_pos)
    df["neg_comp"] = z((z(df.w_vader_neg) + z(df.w_lm_neg) + z(df.w_nrc_negative)) / 3)
    df["pos_comp"] = z((z(df.w_vader_pos) + z(df.w_lm_pos) + z(df.w_nrc_positive)) / 3)
    assert np.allclose(df.neg_comp, df.comp_negative_z) and np.allclose(df.pos_comp, df.comp_positive_z)

    # Positive facial expression (the jangle): means over the presenting founder's frames
    df["joy"] = z(df.face_joy_pf)            # happiness probability from the face-emotion classifier
    df["smile"] = z(df.face_smile_au_pf)     # Duchenne smile: min(AU6, AU12) from OpenGraphAU
    df["pos_face"] = z(df.face_valence_pf)   # positive facial valence, ELIM-FER
    df["pos_face_index"] = z((df.joy + df.smile + df.pos_face) / 3)

    # ---- Interactions: products of the standardized terms
    df["face_x_neg"] = df.face_energy * df.neg_words          # the joint term
    df["face_x_pos"] = df.face_energy * df.pos_words
    df["voice_x_neg"] = df.voice_energy * df.neg_words
    df["verbal_x_neg"] = df.verbal_arousal * df.neg_words
    df["joywords_x_neg"] = df.joy_words * df.neg_words
    df["pfi_x_neg"] = df.pos_face_index * df.neg_words

    # ---- Controls
    # Season fixed effects. Reference = first season in sorted order (S10_2023); any reference
    # gives the same focal estimates, this one reproduces the constants in the tables.
    df["season"] = df.subsegment_id.str.split("-").str[1]
    for s in sorted(df.season.unique())[1:]:
        df[f"ssn_{s}"] = (df.season == s).astype(float)
    # Industry dummies, reference = COMPUTER RELATED ("Technology"). The two pitches without a
    # category get zeros on all dummies, i.e. they enter with the reference category.
    for k in sorted(df.industry_cat.dropna().unique()):
        if k != "COMPUTER RELATED":
            df["ind_" + k.split()[0].split("/")[0].lower()] = (df.industry_cat == k).astype(float)
    # Gender of the founder(s), reference = male
    for g in ["female", "mixed", "unknown"]:
        df[f"gen_{g}"] = (df.gender == g).astype(float)
    return df


def controls(df, industry=True, seasons=True):
    """The standard control set (21 terms): race, founders, speakers, dress, industry, gender, season."""
    base = ["race", "founder_count", "n_active_speakers", "dress_formality"]
    ind = [c for c in df.columns if c.startswith("ind_")]
    gen = ["gen_female", "gen_mixed", "gen_unknown"]
    ssn = [c for c in df.columns if c.startswith("ssn_")]
    return base + (ind if industry else []) + gen + (ssn if seasons else [])


def fit(df, X, y=Y):
    """Logit of y on X plus a constant, SEs clustered by episode (vid_id).

    Columns without variance (possible in subsamples) are dropped."""
    d = df.dropna(subset=[y] + X).copy()
    Xc = sm.add_constant(d[X].astype(float), has_constant="add")
    keep = [c for c in Xc.columns if c == "const" or Xc[c].std() > 0]
    r = sm.Logit(d[y].astype(float), Xc[keep]).fit(
        disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(d.vid_id)[0]})
    return r, d


def fe_fit(df, X, y=Y):
    """Conditional logit with episode fixed effects.

    Episodes without variance in y carry no information and drop out (389 pitches, 98 episodes).
    Returns params, standard errors, covariance matrix, and the estimation sample."""
    from statsmodels.discrete.conditional_models import ConditionalLogit
    d = df.dropna(subset=[y] + X).copy()
    d = d[d.groupby("vid_id")[y].transform("std") > 0]
    Xd = d[X].astype(float)
    keep = [c for c in Xd.columns
            if Xd.groupby(d.vid_id)[c].transform(lambda s: s - s.mean()).abs().sum() > 1e-9]
    r = ConditionalLogit(d[y].astype(float).values, Xd[keep].values,
                         groups=pd.factorize(d.vid_id)[0]).fit(disp=0, method="bfgs", maxiter=2000)
    params, bse = pd.Series(r.params, index=keep), pd.Series(r.bse, index=keep)
    cov = pd.DataFrame(r.cov_params(), index=keep, columns=keep)
    return params, bse, cov, d


def lincom(params, cov, w):
    """sum(w_k * b_k) with delta-method SE and two-sided p, e.g. a simple slope."""
    names = list(w)
    g = np.array([w[n] for n in names], dtype=float)
    b = float(g @ params[names].values)
    se = float(np.sqrt(g @ cov.loc[names, names].values @ g))
    return b, se, float(2 * norm.sf(abs(b / se)))


def pred(r, d, **setv):
    """Average predicted probability over the sample d, with the named variables set to fixed values."""
    X = sm.add_constant(d[[c for c in r.params.index if c != "const"]].astype(float), has_constant="add")
    return float(r.predict(X.assign(**setv)[r.params.index]).mean())


def johnson_neyman(params, cov, mod, f="face_energy", fx="face_x_neg", n_grid=2001):
    """Johnson-Neyman region for the slope of f along the moderator values `mod` (p < .05).

    Returns (share of pitches in which the slope of f is significantly positive,
    threshold on the moderator, share in which it is significantly negative)."""
    mod = np.asarray(mod, dtype=float)
    grid = np.linspace(mod.min(), mod.max(), n_grid)
    bF, bX = params[f], params[fx]
    slope = bF + bX * grid
    se = np.sqrt(cov.loc[f, f] + grid ** 2 * cov.loc[fx, fx] + 2 * grid * cov.loc[f, fx])
    pos, neg = grid[slope / se > 1.959964], grid[slope / se < -1.959964]
    if len(pos) == 0:
        share, thr = 0.0, np.nan
    elif bX > 0:
        thr = pos.min(); share = float((mod >= thr).mean())
    else:
        thr = pos.max(); share = float((mod <= thr).mean())
    nshare = float(((mod >= neg.min()) if bX < 0 else (mod <= neg.max())).mean()) if len(neg) else 0.0
    return share, float(thr), nshare


def thirds(s, labels):
    """Split pitches into thirds by rank: 153, 152, and 152 of the 457 pitches, as in the manuscript.

    Written out instead of pd.qcut because pandas versions place the pitch that sits exactly on a
    tercile edge in different thirds, which moves the raw funding rates by a point."""
    r = s.rank(method="first").values
    n = len(s)
    k1, k2 = (n - 1) // 3 + 1, 2 * (n - 1) // 3 + 1
    return pd.Series(np.where(r <= k1, labels[0], np.where(r <= k2, labels[1], labels[2])), index=s.index)


def converged(r):
    """False if the logit did not converge or some coefficient explodes (|b| > 10), the usual sign
    that a dummy perfectly predicts funding in a small sample (separation)."""
    return bool(r.mle_retvals.get("converged", True)) and float(r.params.abs().max()) <= 10


def coef(r, name):
    """(b, se, p) of one coefficient of a statsmodels result."""
    return float(r.params[name]), float(r.bse[name]), float(r.pvalues[name])


def summarize(r):
    """Compact dump of a fitted model for the results file."""
    return dict(params=r.params.to_dict(), bse=r.bse.to_dict(), p=r.pvalues.to_dict(),
                llf=float(r.llf), prsq=float(r.prsquared), aic=float(r.aic), n=int(r.nobs))


# ---- Formatting (round half up, as in the manuscript)
def rhu(x, nd=2):
    """Round half up (Python's round() rounds half to even, e.g. round(72.5) == 72)."""
    return float(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pct(x):
    """Share as a whole percentage, rounded half up: 0.725 -> 73."""
    return int(rhu(100 * x, 0))


def fp(p):
    """p-value in APA style without the leading zero."""
    if p < .001:
        return "< .001"
    s = f"{rhu(p, 3):.3f}" if p < .01 else f"{rhu(p, 2):.2f}"
    return "= " + s.lstrip("0")


def fb(b, se=None, p=None, nd=2):
    """Coefficient string in the form used in the manuscript, for example '0.31 (0.10), p = .002'."""
    s = f"{rhu(b, nd):.{nd}f}"
    if se is not None:
        s += f" ({rhu(se, nd):.{nd}f})"
    if p is not None:
        s += f", p {fp(p)}"
    return s
