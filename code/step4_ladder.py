"""Step 4. How much does the joint term add? (Table 6)

A ladder of three models: controls only; plus six single-channel cues; plus the one joint
term. In sample: McFadden pseudo R-squared, likelihood-ratio tests, AIC and BIC.
Out of sample: tenfold cross-validation by episode (repeated), and train-on-one-half,
predict-the-other.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

from common import Y, controls, fit, load, summarize

SINGLE = ["face_energy", "voice_energy", "verbal_arousal", "neg_words", "pos_words", "pos_face_index"]
JOINT = ["face_x_neg"]


def _logit_predict(df, X, train, test):
    """Plain (unclustered) logit on the training rows, predicted probabilities for the test rows."""
    Xtr = sm.add_constant(df.loc[train, X].astype(float), has_constant="add")
    keep = [c for c in Xtr.columns if c == "const" or Xtr[c].std() > 0]
    m = sm.Logit(df.loc[train, Y].astype(float), Xtr[keep]).fit(disp=0, maxiter=500, method="bfgs")
    return m.predict(sm.add_constant(df.loc[test, X].astype(float), has_constant="add")[keep])


def cross_validate(df, models, reps, seed=7, folds=10):
    """Tenfold cross-validation with whole episodes as folds, repeated `reps` times; AUC per repetition."""
    eps = df.vid_id.unique()
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pr = {k: np.zeros(len(df)) for k in models}
        for fold in np.array_split(rng.permutation(eps), folds):
            test = df.vid_id.isin(fold).values
            for k, X in models.items():
                pr[k][test] = _logit_predict(df, X, ~test, test)
        out.append({k: roc_auc_score(df[Y], pr[k]) for k in pr})
    return pd.DataFrame(out)


def run(df=None, quick=False):
    df = load() if df is None else df
    C = controls(df)
    models = {"controls": C, "single": C + SINGLE, "joint": C + SINGLE + JOINT}
    fits = {k: fit(df, X)[0] for k, X in models.items()}
    r0, r1, r2 = fits["controls"], fits["single"], fits["joint"]
    lr6, lr1 = 2 * (r1.llf - r0.llf), 2 * (r2.llf - r1.llf)
    R = {"table6": {k: summarize(r) for k, r in fits.items()},
         "prsq": {k: float(r.prsquared) for k, r in fits.items()},
         "lr_single": dict(chi2=float(lr6), df=len(SINGLE), p=float(chi2.sf(lr6, len(SINGLE)))),
         "lr_joint": dict(chi2=float(lr1), df=len(JOINT), p=float(chi2.sf(lr1, len(JOINT)))),
         "aic": {k: float(r.aic) for k, r in fits.items()},
         "bic": {k: float(r.bic) for k, r in fits.items()},
         "n_params": {k: int(len(r.params)) for k, r in fits.items()}}
    A = cross_validate(df, models, reps=5 if quick else 20)
    R["cv_auc_mean"] = A.mean().to_dict()
    R["cv_auc_sd"] = A.std().to_dict()
    R["cv_reps"] = len(A)
    R["cv_joint_beats_single_share"] = float((A.joint > A.single).mean())
    R["cv_single_beats_controls_share"] = float((A.single > A.controls).mean())
    halves = {}
    for tr, te in [("A", "B"), ("B", "A")]:
        train, test = (df.random_half == tr).values, (df.random_half == te).values
        halves[f"{tr}->{te}"] = {k: float(roc_auc_score(df.loc[test, Y], _logit_predict(df, X, train, test)))
                                 for k, X in models.items()}
    R["half_split_auc"] = halves
    return R


if __name__ == "__main__":
    import sys
    from common import fp, rhu
    R = run(quick="--quick" in sys.argv)
    print("Pseudo R2:", {k: rhu(v, 3) for k, v in R["prsq"].items()})
    print(f"LR single cues chi2({R['lr_single']['df']}) = {rhu(R['lr_single']['chi2'], 2)}, p {fp(R['lr_single']['p'])};",
          f"joint term chi2(1) = {rhu(R['lr_joint']['chi2'], 2)}, p {fp(R['lr_joint']['p'])}")
    print("AIC:", {k: rhu(v, 1) for k, v in R["aic"].items()}, "| BIC:", {k: rhu(v, 1) for k, v in R["bic"].items()})
    print("CV AUC:", {k: rhu(v, 2) for k, v in R["cv_auc_mean"].items()}, "| joint > single in",
          f"{rhu(100 * R['cv_joint_beats_single_share'], 0):.0f}% of {R['cv_reps']} repetitions")
    print("Half split:", {k: {m: rhu(a, 2) for m, a in v.items()} for k, v in R["half_split_auc"].items()})
