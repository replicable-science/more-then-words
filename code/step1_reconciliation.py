"""Step 1. Contradiction -> reconciliation, made possible by separability (Table 2).

Two debates meet in one model. Delivery: should founders display passion (facial energy)?
Content: should they name problems (negative words)? Measured apart, in the same pitches,
the two cues interact, and each settles the other debate.
"""
from common import Y, controls, fit, lincom, load, pred, summarize, coef, thirds


def run(df=None, quick=False):
    df = load() if df is None else df
    C = controls(df)
    F = ["face_energy", "voice_energy", "neg_words"]
    models = {
        "M1": C,
        "M2": C + F,                                          # each cue on its own
        "M3": C + F + ["face_x_neg"],                         # the joint test
        "M4": C + F + ["face_x_neg", "pos_words", "face_x_pos"],   # is it negativity or valence?
        "M5": C + ["face_energy", "voice_energy", "verbal_arousal", "neg_words",
                   "face_x_neg", "voice_x_neg", "verbal_x_neg"],     # which channel carries it?
    }
    fits = {k: fit(df, X) for k, X in models.items()}
    R = {"table2": {k: summarize(r) for k, (r, _) in fits.items()}}

    r2, _ = fits["M2"]
    r3, d3 = fits["M3"]
    cov = r3.cov_params()
    R["m2_face"], R["m2_neg"] = coef(r2, "face_energy"), coef(r2, "neg_words")
    R["m3_face_x_neg"] = coef(r3, "face_x_neg")
    R["m4_face_x_pos"] = coef(fits["M4"][0], "face_x_pos")
    R["m5_voice_x_neg"] = coef(fits["M5"][0], "voice_x_neg")
    R["m5_verbal_x_neg"] = coef(fits["M5"][0], "verbal_x_neg")

    # Read 1 (delivery debate): slope of facial energy in upbeat vs problem-focused pitches.
    # Calm / animated = 1 SD below / above the mean of facial energy.
    for lab, q in [("p10", .10), ("p90", .90)]:
        g = d3.neg_words.quantile(q)
        R[f"face_slope_{lab}"] = lincom(r3.params, cov, {"face_energy": 1, "face_x_neg": g})
        R[f"pred_{lab}_calm"] = pred(r3, d3, face_energy=-1, neg_words=g, face_x_neg=-g)
        R[f"pred_{lab}_animated"] = pred(r3, d3, face_energy=1, neg_words=g, face_x_neg=g)

    # Read 2 (content debate): slope of negative words for calm vs animated founders,
    # with predicted funding at few (-1 SD) and many (+1 SD) negative words.
    for lab, fv in [("calm", -1), ("animated", 1)]:
        R[f"neg_slope_{lab}"] = lincom(r3.params, cov, {"neg_words": 1, "face_x_neg": fv})
        R[f"pred_{lab}_few_neg"] = pred(r3, d3, face_energy=fv, neg_words=-1, face_x_neg=-fv)
        R[f"pred_{lab}_many_neg"] = pred(r3, d3, face_energy=fv, neg_words=1, face_x_neg=fv)

    # The same story without a model: raw funding rates by thirds of both cues
    t_face = thirds(df.face_energy, ["calm", "mid", "animated"])
    t_neg = thirds(df.neg_words, ["few", "mid", "many"])
    rates = df.groupby([t_neg, t_face])[Y].agg(["mean", "size"])
    R["raw_rates"] = {f"{n}_{f}": dict(rate=float(rates.loc[(n, f), "mean"]), n=int(rates.loc[(n, f), "size"]))
                      for n in ["few", "mid", "many"] for f in ["calm", "mid", "animated"]}
    R["funding_rate"] = float(df[Y].mean())
    R["n"], R["episodes"], R["funded"] = len(df), int(df.vid_id.nunique()), int(df[Y].sum())
    R["n_params_m3"] = int(len(r3.params))
    return R


if __name__ == "__main__":
    from common import fb, pct
    R = run()
    print("Model 2 (on average): facial energy", fb(*R["m2_face"]), "| negative words", fb(*R["m2_neg"]))
    print("Model 3 (joint test): facial energy x negative words", fb(*R["m3_face_x_neg"]))
    for lab in ["p90", "p10"]:
        print(f"  slope of facial energy at {lab} of negative words: {fb(*R['face_slope_' + lab])};",
              f"predicted funding calm {pct(R[f'pred_{lab}_calm'])}% vs animated {pct(R[f'pred_{lab}_animated'])}%")
    for lab in ["calm", "animated"]:
        print(f"  slope of negative words for {lab} founders: {fb(*R['neg_slope_' + lab])};",
              f"few {pct(R[f'pred_{lab}_few_neg'])}% -> many {pct(R[f'pred_{lab}_many_neg'])}%")
    rr = R["raw_rates"]
    print("Raw rates, many negative words: calm", pct(rr["many_calm"]["rate"]), "animated", pct(rr["many_animated"]["rate"]),
          "| few negative words: calm", pct(rr["few_calm"]["rate"]), "animated", pct(rr["few_animated"]["rate"]))
    print("Model 4 face x positive words", fb(*R["m4_face_x_pos"]), "| Model 5 voice x neg", fb(*R["m5_voice_x_neg"]),
          "| verbal x neg", fb(*R["m5_verbal_x_neg"]))
