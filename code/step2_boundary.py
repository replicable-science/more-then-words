"""Step 2. Unclear scope -> boundary specification, made possible by resolution (Table 3).

A Johnson-Neyman analysis of the joint test (Table 2, Model 3) locates the share of pitches
in which displayed passion pays significantly. A cluster bootstrap shows how certain that
boundary is, and random subsets of episodes show how many pitches it takes to see it.
"""
import time

import numpy as np
import pandas as pd

from common import Y, Y2, controls, converged, fe_fit, fit, johnson_neyman, lincom, load, pred, z

FOCAL = ["face_energy", "voice_energy", "neg_words", "face_x_neg"]


def bootstrap(d3, X, reps, seed=2026):
    """Cluster bootstrap by episode: resample episodes with replacement, refit, redo the JN analysis."""
    rng = np.random.default_rng(seed)
    eps = d3.vid_id.unique()
    by_ep = {e: d3[d3.vid_id == e] for e in eps}
    shares, thr_pcts, coefs, conv = [], [], [], []
    for _ in range(reps):
        parts = []
        for j, e in enumerate(rng.choice(eps, len(eps), replace=True)):
            p_ = by_ep[e].copy()
            p_["vid_id"] = f"{e}_{j}"   # a resampled episode counts as its own cluster
            parts.append(p_)
        db = pd.concat(parts, ignore_index=True)
        try:
            rb, ddb = fit(db, X)
            s_, t_, _ = johnson_neyman(rb.params, rb.cov_params(), ddb.neg_words.values)
            shares.append(s_)
            coefs.append(float(rb.params["face_x_neg"]))
            conv.append(converged(rb))
            # locate the bootstrap threshold in the ORIGINAL distribution of negative words
            thr_pcts.append(float((d3.neg_words < t_).mean()) if not np.isnan(t_) else np.nan)
        except Exception:
            pass
    S, P, B = np.array(shares), np.array(thr_pcts), np.array(coefs)
    return dict(reps=len(S), share_median=float(np.median(S)),
                share_lo=float(np.percentile(S, 2.5)), share_hi=float(np.percentile(S, 97.5)),
                thr_pct_median=float(np.nanmedian(P)), thr_pct_lo=float(np.nanpercentile(P, 2.5)),
                thr_pct_hi=float(np.nanpercentile(P, 97.5)), coef_lo=float(np.percentile(B, 2.5)),
                coef_hi=float(np.percentile(B, 97.5)), coef_positive_share=float((B > 0).mean()),
                no_region_share=float(np.isnan(P).mean()),   # draws without any significant region
                not_converged=int(len(conv) - sum(conv)))


def subsets(df, X, reps, targets=(100, 150, 230, 300), seed=11):
    """Random subsets of whole episodes of about `target` pitches: how often is the interaction
    significantly positive, and how often is there any region in which passion pays significantly?"""
    rng = np.random.default_rng(seed)
    eps = df.vid_id.unique()
    sizes = df.groupby("vid_id").size()
    out = {}
    for target in targets:
        sig = reg = ok = clean = sig_clean = 0
        ns = []
        for _ in range(reps):
            perm = rng.permutation(eps)
            take = perm[: int((sizes[perm].cumsum() < target).sum()) + 1]
            try:
                r, dd = fit(df[df.vid_id.isin(take)], X)
                if not np.isfinite(r.bse["face_x_neg"]):
                    continue
                ok += 1
                ns.append(len(dd))
                s_ = (r.pvalues["face_x_neg"] < .05) and (r.params["face_x_neg"] > 0)
                sig += s_
                reg += johnson_neyman(r.params, r.cov_params(), dd.neg_words.values)[0] > 0
                if converged(r):          # separation-free fits only (version-independent)
                    clean += 1
                    sig_clean += s_
            except Exception:
                pass
        out[str(target)] = dict(n_mean=float(np.mean(ns)), reps=ok, share_significant=sig / ok, share_with_region=reg / ok,
                                failed=reps - ok, not_converged=ok - clean, share_significant_converged=sig_clean / clean)
    return out


def gain_curve(r3, d3, draws, seed=2026):
    """Funding gain from a passionate face (+1 SD vs -1 SD), in percentage points, along the
    percentiles of negative words, with a 95% band from parametric simulation draws."""
    pcts = np.arange(2, 99)
    g = np.quantile(d3.neg_words.values, pcts / 100)
    names = list(r3.params.index)
    X = d3[[c for c in names if c != "const"]].astype(float).assign(const=1.0)[names]
    rng = np.random.default_rng(seed)
    B = np.column_stack([r3.params.values, rng.multivariate_normal(r3.params.values, r3.cov_params().values, draws).T])
    curves = []
    for gg in g:   # columns of B: the point estimate first, then the simulation draws
        hi = X.assign(face_energy=1.0, neg_words=gg, face_x_neg=gg).values @ B
        lo = X.assign(face_energy=-1.0, neg_words=gg, face_x_neg=-gg).values @ B
        curves.append((1 / (1 + np.exp(-hi)) - 1 / (1 + np.exp(-lo))).mean(axis=0))
    curves = np.array(curves)                     # percentiles x (1 + draws)
    pt, sims = curves[:, 0], curves[:, 1:]
    lo, hi = np.percentile(sims, 2.5, axis=1), np.percentile(sims, 97.5, axis=1)
    first = int(pcts[np.argmax(lo > 0)]) if (lo > 0).any() else None
    return dict(pct=pcts.tolist(), gain=pt.tolist(), lo=lo.tolist(), hi=hi.tolist(), band_excludes_zero_from_pct=first)


def run(df=None, quick=False):
    df = load() if df is None else df
    C = controls(df)
    X = C + FOCAL
    r3, d3 = fit(df, X)
    cov = r3.cov_params()
    R = {}

    # ---- Johnson-Neyman region in the full corpus (Table 3, Panel A; Figure 4)
    share, thr, nshare = johnson_neyman(r3.params, cov, d3.neg_words.values)
    R["jn"] = dict(threshold_z=thr, threshold_pct=float((d3.neg_words < thr).mean()), share_positive=share, share_negative=nshare)
    rows = []
    for lab, g in [("10th", d3.neg_words.quantile(.10)), ("25th", d3.neg_words.quantile(.25)),
                   ("50th", d3.neg_words.quantile(.50)), ("threshold", thr),
                   ("75th", d3.neg_words.quantile(.75)), ("90th", d3.neg_words.quantile(.90))]:
        b, se, p = lincom(r3.params, cov, {"face_energy": 1, "face_x_neg": g})
        rows.append(dict(label=lab, neg_words_z=float(g), slope=b, se=se, p=p,
                         pred_calm=pred(r3, d3, face_energy=-1, neg_words=g, face_x_neg=-g),
                         pred_animated=pred(r3, d3, face_energy=1, neg_words=g, face_x_neg=g)))
    R["table3_panelA"] = rows
    # Predicted funding beyond and below the threshold (negative words kept at observed values)
    for lab, dd in [("above", d3[d3.neg_words >= thr]), ("below", d3[d3.neg_words < thr])]:
        R[f"{lab}_threshold"] = dict(
            n=len(dd), raw_rate=float(dd[Y].mean()),
            pred_calm=pred(r3, dd, face_energy=-1, face_x_neg=-dd.neg_words),
            pred_animated=pred(r3, dd, face_energy=1, face_x_neg=dd.neg_words))

    # ---- The region in the random halves, with episode fixed effects, and for the second outcome
    for h in ["A", "B"]:
        r, d = fit(df[df.random_half == h], X)
        s, t, n_ = johnson_neyman(r.params, r.cov_params(), d.neg_words.values)
        R[f"half_{h}"] = dict(b=float(r.params["face_x_neg"]), p=float(r.pvalues["face_x_neg"]), jn_share=s,
                              n=len(d), episodes=int(d.vid_id.nunique()), converged=converged(r))
    params, bse, covf, dfe = fe_fit(df, controls(df, seasons=False) + FOCAL)
    R["fe"] = dict(b=float(params["face_x_neg"]), se=float(bse["face_x_neg"]),
                   jn_share=johnson_neyman(params, covf, dfe.neg_words.values)[0], n=len(dfe), episodes=int(dfe.vid_id.nunique()))
    rp, dp = fit(df, X, y=Y2)
    R["pitch_success"] = dict(b=float(rp.params["face_x_neg"]), p=float(rp.pvalues["face_x_neg"]),
                              jn_share=johnson_neyman(rp.params, rp.cov_params(), dp.neg_words.values)[0])

    # ---- Resolution in scale: bootstrap (Panel B) and random subsets (Panel C)
    t0 = time.time()
    R["bootstrap"] = bootstrap(d3, X, reps=100 if quick else 1000)
    R["subsets"] = subsets(df, X, reps=50 if quick else 400)
    R["seconds_resampling"] = round(time.time() - t0, 1)

    # ---- Extra checks, not reported in the manuscript
    # (a) Point where the estimated slope of facial energy crosses zero, vs. the JN threshold
    g0 = float(-r3.params["face_energy"] / r3.params["face_x_neg"])
    R["extra_zero_crossing"] = dict(neg_words_z=g0, pct=float((d3.neg_words < g0).mean()))
    # (b) The boundary on the probability scale (gain in percentage points, simulation band)
    R["extra_gain_curve"] = gain_curve(r3, d3, draws=300 if quick else 2000)
    # (c) Resolution in time: facial energy measured in each third of the pitch x negative words
    thirds = {}
    for t in ["t1", "t2", "t3"]:
        dt = df.copy()
        dt["face_t"] = z(dt[f"arousal_visual_norm_{t}"])
        dt["face_t_x_neg"] = dt.face_t * dt.neg_words
        r, _ = fit(dt, C + ["face_t", "voice_energy", "neg_words", "face_t_x_neg"])
        thirds[t] = dict(b=float(r.params["face_t_x_neg"]), se=float(r.bse["face_t_x_neg"]), p=float(r.pvalues["face_t_x_neg"]))
    R["extra_time_thirds"] = thirds
    R["extra_face_thirds_corr"] = df[[f"arousal_visual_norm_{t}" for t in ["t1", "t2", "t3"]]].corr().values.tolist()
    return R


if __name__ == "__main__":
    import sys
    from common import fb, pct
    R = run(quick="--quick" in sys.argv)
    j = R["jn"]
    print(f"JN threshold at the {pct(j['threshold_pct'])}th percentile; passion pays significantly in {pct(j['share_positive'])}%"
          f" of pitches; significantly negative in {pct(j['share_negative'])}%")
    a, b = R["above_threshold"], R["below_threshold"]
    print(f"Beyond the threshold: calm {pct(a['pred_calm'])}% vs animated {pct(a['pred_animated'])}%;"
          f" below: {pct(b['pred_calm'])}% vs {pct(b['pred_animated'])}%")
    bs = R["bootstrap"]
    print(f"Bootstrap ({bs['reps']} reps): coefficient 95% interval {bs['coef_lo']:.2f} to {bs['coef_hi']:.2f};"
          f" positive in {pct(bs['coef_positive_share'])}%; threshold between the {pct(bs['thr_pct_lo'])}th and {pct(bs['thr_pct_hi'])}th percentile")
    print("Subsets:", {k: pct(v["share_significant"]) for k, v in R["subsets"].items()})
    print("Halves JN share:", pct(R["half_A"]["jn_share"]), pct(R["half_B"]["jn_share"]), "| FE", fb(R["fe"]["b"], R["fe"]["se"]),
          "| pitch success", fb(R["pitch_success"]["b"], p=R["pitch_success"]["p"]))
    print("Extra: zero crossing at the", pct(R["extra_zero_crossing"]["pct"]), "th percentile; probability band excludes zero from the",
          R["extra_gain_curve"]["band_excludes_zero_from_pct"], "th percentile")
    print("Extra: thirds", {k: fb(v["b"], v["se"], v["p"]) for k, v in R["extra_time_thirds"].items()})
