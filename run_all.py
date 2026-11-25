"""Run all six steps, write output/results.json and output/results.md, and check every
number the manuscript reports against this run.

    python run_all.py           full run (about 2 to 4 minutes)
    python run_all.py --quick   fewer bootstrap, subset and cross-validation repetitions (about 1 minute);
                                the checks that depend on those repetitions are then skipped
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))

import numpy as np  # noqa: E402

import read_examples, step1_reconciliation, step2_boundary, step3_pruning, step4_ladder, step5_robustness, step6_figures  # noqa: E402
from common import OUT, load, pct, rhu  # noqa: E402


# ---------------------------------------------------------------- formatting as in the manuscript
def f1(x): return f"{rhu(x, 1):.1f}"
def f2(x): return f"{rhu(x, 2):.2f}".replace("-0.00", "0.00")
def r2(x): return f"{rhu(x, 2):.2f}".replace("0.", ".", 1).replace("-.00", ".00")     # correlations: .12, -.01
def r3(x): return f"{rhu(x, 3):.3f}".replace("0.", ".", 1)                           # pseudo R2: .038
def p2(p): return "= " + f"{rhu(p, 2):.2f}".lstrip("0")
def p3(p): return "= " + f"{rhu(p, 3):.3f}".lstrip("0")
def pp(p): return "< .001" if p < .001 else p3(p)
def pc(x): return f"{pct(x)}%"
def ordp(x):
    n = pct(x)
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


LAB = {"face_energy": "facial energy", "voice_energy": "vocal energy", "verbal_arousal": "verbal arousal",
       "joy_words": "joyful words", "joy": "joy", "smile": "smile", "pos_face": "positive face",
       "pos_face_index": "positive-face index", "pfi_x_neg": "index x negative words"}
def lab(k): return " with ".join(LAB.get(x, x) for x in k.split("|"))


# ---------------------------------------------------------------- every number the manuscript reports
# (where, what, as reported, value in this run, depends on resampling?)
def manuscript_checks(R):
    s1, s2, s3, s4, s5 = (R[k] for k in ["step1", "step2", "step3", "step4", "step5"])
    rr = s1["raw_rates"]
    t5 = {x["label"]: x for x in s5["table5"]}
    C = []
    add = lambda where, what, reported, run, resampling=False: C.append(dict(where=where, what=what, reported=reported, run=run, resampling=resampling))
    W = "Pipeline at Work"
    add(W, "Pitches, episodes", "457, 116", f"{s1['n']}, {s1['episodes']}")
    add(W, "Share funded", "34%", pc(s1["funding_rate"]))
    add(W, "Share with pitch success (second outcome)", "65%", pc(R["meta"]["pitch_success_rate"]))
    add(W, "Correlations of the three negative-word dictionaries", ".55 to .66", s5["_r_neg_range"])
    add(W, "Verbal arousal: alpha across thirds", ".44", r2(s3["extra_alpha_thirds"]["verbal_arousal"]))
    W = "Contradiction"
    add(W, "Facial energy on average (Table 2, M2): b", "0.23", f2(s1["m2_face"][0]))
    add(W, "Facial energy on average (Table 2, M2): p", "= .08", p2(s1["m2_face"][2]))
    add(W, "Negative words on average (Table 2, M2): b", "-0.26", f2(s1["m2_neg"][0]))
    add(W, "Negative words on average (Table 2, M2): p", "= .015", p3(s1["m2_neg"][2]))
    add(W, "Facial energy x negative words (Table 2, M3): b", "0.31", f2(s1["m3_face_x_neg"][0]))
    add(W, "Facial energy x negative words (Table 2, M3): p", "= .002", pp(s1["m3_face_x_neg"][2]))
    add(W, "Slope of facial energy, 90th pct of negative words: b", "0.63", f2(s1["face_slope_p90"][0]))
    add(W, "Slope of facial energy, 90th pct: p", "< .001", pp(s1["face_slope_p90"][2]))
    add(W, "Predicted funding at 90th pct: calm, animated", "15%, 36%", f"{pc(s1['pred_p90_calm'])}, {pc(s1['pred_p90_animated'])}")
    add(W, "Slope of facial energy, 10th pct: b", "-0.04", f2(s1["face_slope_p10"][0]))
    add(W, "Slope of facial energy, 10th pct: p", "= .77", p2(s1["face_slope_p10"][2]))
    add(W, "Predicted funding at 10th pct: calm, animated", "41%, 39%", f"{pc(s1['pred_p10_calm'])}, {pc(s1['pred_p10_animated'])}")
    add(W, "Raw rates, most negative third: calm, animated", "22%, 36%", f"{pc(rr['many_calm']['rate'])}, {pc(rr['many_animated']['rate'])}")
    add(W, "Raw rates, least negative third: calm, animated", "34%, 33%", f"{pc(rr['few_calm']['rate'])}, {pc(rr['few_animated']['rate'])}")
    add(W, "Slope of negative words, calm face: b", "-0.67", f2(s1["neg_slope_calm"][0]))
    add(W, "Slope of negative words, calm face: p", "< .001", pp(s1["neg_slope_calm"][2]))
    add(W, "Calm face, few to many negative words", "43% to 17%", f"{pc(s1['pred_calm_few_neg'])} to {pc(s1['pred_calm_many_neg'])}")
    add(W, "Slope of negative words, animated face: b", "-0.06", f2(s1["neg_slope_animated"][0]))
    add(W, "Slope of negative words, animated face: p", "= .63", p2(s1["neg_slope_animated"][2]))
    add(W, "Animated face, few to many negative words", "39% to 36%", f"{pc(s1['pred_animated_few_neg'])} to {pc(s1['pred_animated_many_neg'])}")
    rob = ["Random half A", "Random half B", "Episode fixed effects", "Second outcome: pitch success", "Negative words: VADER",
           "Negative words: NRC", "Negative words: average of the three dictionaries", "Word vectors: low verbal enthusiasm",
           "Facial arousal net of mouth opening", "Adding content controls"]
    maxp = max(t5[k]["p"] for k in rob)
    add(W, "Interaction holds in the ten checks listed in the text (Table 5)", "all p < .05", "all p < .05" if maxp < .05 else f"max p {p3(maxp)}")
    add(W, "Facial energy x positive words (Table 2, M4)", "-0.06, p = .54", f"{f2(s1['m4_face_x_pos'][0])}, p {p2(s1['m4_face_x_pos'][2])}")
    add(W, "Vocal energy x negative words (Table 2, M5)", "-0.10, p = .18", f"{f2(s1['m5_voice_x_neg'][0])}, p {p2(s1['m5_voice_x_neg'][2])}")
    add(W, "Verbal arousal x negative words (Table 2, M5)", "0.04, p = .72", f"{f2(s1['m5_verbal_x_neg'][0])}, p {p2(s1['m5_verbal_x_neg'][2])}")
    add(W, "Positive-face index x negative words (Table 4, M6)", "0.00, p = .96", f"{f2(s3['index_x_neg'][0])}, p {p2(s3['index_x_neg'][2])}")
    W = "Unbounded Scope"
    jn = s2["jn"]
    add(W, "Johnson-Neyman threshold (percentile of negative words)", "65th", ordp(jn["threshold_pct"]))
    add(W, "Share of pitches in which passion pays significantly", "35%", pc(jn["share_positive"]))
    add(W, "Share in which passion costs significantly", "0%", pc(jn["share_negative"]))
    a, b = s2["above_threshold"], s2["below_threshold"]
    add(W, "Beyond the threshold: calm, animated", "19%, 38%", f"{pc(a['pred_calm'])}, {pc(a['pred_animated'])}")
    add(W, "Below the threshold: calm, animated", "36%, 37%", f"{pc(b['pred_calm'])}, {pc(b['pred_animated'])}")
    bs = s2["bootstrap"]
    add(W, "Bootstrap: interaction positive in", "994 of 1,000", f"{int(round(bs['coef_positive_share'] * bs['reps'])):,} of {bs['reps']:,}", True)
    add(W, "Bootstrap: 95% interval of the interaction", "0.08 to 0.57", f"{f2(bs['coef_lo'])} to {f2(bs['coef_hi'])}", True)
    add(W, "Bootstrap: threshold between percentiles", "38th and 92nd", f"{ordp(bs['thr_pct_lo'])} and {ordp(bs['thr_pct_hi'])}", True)
    sub = s2["subsets"]
    for k, rep, txt in [("100", "19%", "one of five"), ("150", "28%", "Table 3, Panel C"), ("230", "61%", "nearly three of four"),
                        ("300", "78%", "nine of ten")]:
        add(W, f"Subsets of about {k} pitches: interaction significant in ({txt})", rep, pc(sub[k]["share_significant"]), True)
    add(W, "Region in which passion pays, random halves A and B", "16% and 34%", f"{pc(s2['half_A']['jn_share'])} and {pc(s2['half_B']['jn_share'])}")
    W = "Redundancy"
    jr, gr = s3["jingle_r"], s3["jangle_r"]
    add(W, "Jingle: facial energy with vocal energy", ".12", r2(jr["face_energy|voice_energy"][0]))
    add(W, "Jingle: facial energy with joyful words", "-.01", r2(jr["face_energy|joy_words"][0]))
    add(W, "Jingle: vocal energy with joyful words", ".05", r2(jr["voice_energy|joy_words"][0]))
    add(W, "Jangle: joy with smile", ".65", r2(gr["joy|smile"]))
    add(W, "Jangle: joy with positive valence", ".81", r2(gr["joy|pos_face"]))
    add(W, "Jangle: smile with positive valence", ".69", r2(gr["smile|pos_face"]))
    add(W, "Jangle: alpha of the three", ".88", r2(s3["alpha_pos_face_index"]))
    for key, what, rep in [("joy", "Joy alone (Table 4, M1)", "0.01, p = .89"), ("smile", "Smile alone (M2)", "0.05, p = .66"),
                           ("pos_face", "Positive face alone (M3)", "0.15, p = .19")]:
        v = s3["alone"][key]; add(W, what, rep, f"{f2(v[0])}, p {p2(v[2])}")
    v = s3["together"]
    add(W, "Together (M4): joy", "-0.27, p = .12", f"{f2(v['joy'][0])}, p {p2(v['joy'][2])}")
    add(W, "Together (M4): positive face", "0.39, p = .050", f"{f2(v['pos_face'][0])}, p {p3(v['pos_face'][2])}")
    add(W, "Positive-face index (M5)", "0.06, p = .58", f"{f2(s3['index'][0])}, p {p2(s3['index'][2])}")
    sq = s3["squared_p"]
    add(W, "Inverted U, squared terms: p (joy, smile, positive face, index)", ".13, .88, .08, .19",
        ", ".join(p2(sq[k])[2:] for k in ["joy", "smile", "pos_face", "pos_face_index"]))
    m2 = s3["second_outcome_min_p"]
    add(W, "Second outcome, positive-expression terms", "all p > .12", "all p > .12" if m2 > .12 else f"min p {p3(m2)}")
    mx = max(abs(x) for x in s3["r_face_energy_with_positive_expression"].values())
    add(W, "Facial energy with positive expression (absolute r)", "r ≤ .08", f"r ≤ {r2(mx)}")
    jw = s3["joy_words_raw"]
    add(W, "Table 4 note: joyful words M, SD", "2.76, 1.77", f"{f2(jw['mean'])}, {f2(jw['sd'])}")
    add(W, "Table 4 note: joyful words on funding", "0.13, p = .23", f"{f2(s3['joy_words_alone'][0])}, p {p2(s3['joy_words_alone'][2])}")
    add(W, "Table 4 note: joyful words x negative words", "0.04, p = .72", f"{f2(s3['joy_words_x_neg'][0])}, p {p2(s3['joy_words_x_neg'][2])}")
    h = s3["jangle_r_halves"]
    for key, rep in [("joy|smile", ".65 and .66"), ("joy|pos_face", ".80 and .81"), ("smile|pos_face", ".69 and .70")]:
        lo_, hi_ = sorted([h["A"][key], h["B"][key]])
        add(W, f"Table 4 note: {lab(key)} in the two halves", rep, f"{r2(lo_)} and {r2(hi_)}")
    W = "Programmatic Account"
    pr = s4["prsq"]
    add(W, "McFadden pseudo R2: controls, + six cues, + joint term", ".038, .060, .074", ", ".join(r3(pr[k]) for k in ["controls", "single", "joint"]))
    add(W, "Gain in pseudo R2: joint term against six cues", ".014 against .022",
        f"{r3(pr['joint'] - pr['single'])} against {r3(pr['single'] - pr['controls'])}")
    add(W, "LR test, six single cues", "chi2(6) = 13.05, p = .04", f"chi2({s4['lr_single']['df']}) = {f2(s4['lr_single']['chi2'])}, p {p2(s4['lr_single']['p'])}")
    add(W, "LR test, joint term", "chi2(1) = 8.12, p = .004", f"chi2(1) = {f2(s4['lr_joint']['chi2'])}, p {p3(s4['lr_joint']['p'])}")
    add(W, "AIC: controls, + six cues, + joint term", "608.7, 607.7, 601.5", ", ".join(f1(s4["aic"][k]) for k in ["controls", "single", "joint"]))
    cv = s4["cv_auc_mean"]
    add(W, "Cross-validated AUC: controls, + six cues, + joint term", ".51, .53, .55", ", ".join(r2(cv[k]) for k in ["controls", "single", "joint"]), True)
    add(W, "Joint term beats six cues in every repetition", "20 of 20", f"{int(round(s4['cv_joint_beats_single_share'] * s4['cv_reps']))} of {s4['cv_reps']}", True)
    hs = s4["half_split_auc"]
    better = [k for k, v in hs.items() if rhu(v["joint"], 2) > rhu(v["single"], 2)]
    same = [k for k, v in hs.items() if rhu(v["joint"], 2) == rhu(v["single"], 2)]
    add(W, "Half split: one direction improves", ".54 to .55", f"{r2(hs[better[0]]['single'])} to {r2(hs[better[0]]['joint'])}" if better else "no improvement")
    add(W, "Half split: the other direction", ".55 with and without", f"{r2(hs[same[0]]['single'])} with and without" if same else "differs")
    return C


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def compare_tables(R):
    """Max absolute differences against the unrounded values behind Tables 2, 4, 5 and 6."""
    ref = json.loads((ROOT / "reference" / "manuscript_tables.json").read_text())
    rows = []
    for tab, run_tab in [("table2", R["step1"]["table2"]), ("table4_panelB", R["step3"]["table4_panelB"]), ("table6", R["step4"]["table6"])]:
        for m, refm in ref[tab].items():
            got = run_tab[m]
            dp = max(abs(got["params"][k] - v) for k, v in refm["params"].items())
            ds = max(abs(got["bse"][k] - v) for k, v in refm["bse"].items())
            shown = {"table2": "Table 2", "table4_panelB": "Table 4, Panel B", "table6": "Table 6"}
            rows.append((f"{shown[tab]} {m}", len(refm["params"]), dp, ds, abs(got["llf"] - refm["llf"])))
    got5 = {x["label"]: x for x in R["step5"]["table5"]}
    d5 = max(max(abs(got5[x["label"]][k] - x[k]) for k in ["b", "se", "p", "jn_share"]) for x in ref["table5"] if x["label"] in got5)
    rows.append(("Table 5 (14 rows)", 4 * 14, d5, d5, 0.0))
    return rows


def write_markdown(R, checks, tables, quick):
    s1, s2, s3, s4, s5 = (R[k] for k in ["step1", "step2", "step3", "step4", "step5"])
    L = ["# Results of this run", "",
         f"Run on {time.strftime('%Y-%m-%d %H:%M')}{' in QUICK mode (fewer repetitions)' if quick else ''}; "
         f"{R['meta']['seconds']:.0f} seconds. All numbers are recomputed from `data/pitches.csv`.", ""]
    ok = [c for c in checks if not (quick and c["resampling"])]
    bad = [c for c in ok if c["reported"].split(" (")[0] != c["run"]]
    L += ["## 1. Check against the manuscript", "",
          f"{len(ok) - len(bad)} of {len(ok)} reported numbers reproduced exactly (as rounded in the text)"
          + (f"; {len(checks) - len(ok)} resampling-based checks skipped in quick mode" if quick else "") + ".", ""]
    if bad:
        L += ["**Mismatches:**", ""] + [f"- {c['where']}: {c['what']}: manuscript **{c['reported']}**, this run **{c['run']}**" for c in bad] + [""]
    L += ["| Section | Number | Manuscript | This run | Same |", "|---|---|---|---|---|"]
    for c in checks:
        same = "skipped (quick)" if (quick and c["resampling"]) else ("yes" if c["reported"].split(" (")[0] == c["run"] else "**NO**")
        L.append(f"| {c['where']} | {c['what']} | {c['reported']} | {c['run']} | {same} |")
    L += ["", "Unrounded tables against the values behind the manuscript tables (max absolute difference):", "",
          "| Table, model | Terms | Coefficients | SEs | Log likelihood |", "|---|---|---|---|---|"]
    L += [f"| {t} | {n} | {dp:.1e} | {ds:.1e} | {dl:.1e} |" for t, n, dp, ds, dl in tables]

    L += ["", "## 2. Additional checks (not reported as manuscript numbers)", ""]
    zc, gc = s2["extra_zero_crossing"], s2["extra_gain_curve"]
    L += [          f"- **Zero crossing vs. significance threshold.** The estimated slope of facial energy turns positive at the "
          f"{ordp(zc['pct'])} percentile of negative words; it becomes significant (Johnson-Neyman) at the {ordp(s2['jn']['threshold_pct'])}.",
          f"- **Boundary on the probability scale.** The 95% simulation band of the funding gain excludes zero from the "
          f"{ordp(gc['band_excludes_zero_from_pct'])} percentile on (log-odds scale: {ordp(s2['jn']['threshold_pct'])}).",
          f"- **Bootstrap draws without a region.** In {pc(s2['bootstrap']['no_region_share'])} of the {s2['bootstrap']['reps']:,} bootstrap draws, "
          f"passion pays significantly nowhere; the {ordp(s2['bootstrap']['thr_pct_lo'])}-to-{ordp(s2['bootstrap']['thr_pct_hi'])} interval for the threshold refers to the other draws. Median share of "
          f"pitches in the region: {pc(s2['bootstrap']['share_median'])} (95% interval {pc(s2['bootstrap']['share_lo'])} to {pc(s2['bootstrap']['share_hi'])}).",
          "- **Small subsets and separation.** Logit fits that did not converge or have a coefficient beyond 10 in absolute value "
          "(a dummy perfectly predicts funding), out of 400 subsets each: "
          + "; ".join(f"about {k} pitches {v['not_converged'] + v['failed']}" for k, v in s2["subsets"].items())
          + ". Share significant among the clean fits only: "
          + ", ".join(f"{pc(v['share_significant_converged'])}" for v in s2["subsets"].values())
          + f" (all fits: {', '.join(pc(v['share_significant']) for v in s2['subsets'].values())}). Older statsmodels versions (0.14) "
          "stop with an error on some of these fits and drop them, which moves two of the all-fits shares by one point. "
          f"Bootstrap draws that did not converge: {s2['bootstrap']['not_converged']} of {s2['bootstrap']['reps']:,}; "
          f"both random-half fits converged: {'yes' if s2['half_A']['converged'] and s2['half_B']['converged'] else 'no'}.",
          "- **Resolution in time.** Facial energy measured in each third of the pitch, times negative words: "
          + "; ".join(f"{dict(t1='first', t2='second', t3='last')[t]} third b = {f2(v['b'])} ({f2(v['se'])}), p {pp(v['p'])}" for t, v in s2["extra_time_thirds"].items())
          + ". Facial energy correlates " + ", ".join(r2(s2["extra_face_thirds_corr"][i][j]) for i, j in [(0, 1), (0, 2), (1, 2)])
          + " across thirds.",
          "- **Is the jingle just unreliability?** Alpha across thirds: " + ", ".join(f"{lab(k)} {r2(v)}" for k, v in s3["extra_alpha_thirds"].items())
          + ". Jingle correlations corrected for attenuation: " + ", ".join(f"{lab(k)} {r2(v)}" for k, v in s3["extra_jingle_disattenuated"].items()) + ".",
          "- **How large could the pruned effects be?** 95% confidence intervals (log-odds per SD): "
          + "; ".join(f"{lab(k)} [{f2(v[0])}, {f2(v[1])}]" for k, v in s3["extra_ci95"].items()) + ".",
          f"- **Events per parameter.** {s1['funded']} funded pitches, {s1['n_params_m3']} parameters in Table 2, Model 3 "
          f"({s1['funded'] / s1['n_params_m3']:.1f} events per parameter). Sparse controls (no industry dummies, no season effects, "
          f"{s5['extra_sparse']['n_params']} parameters): interaction b = {f2(s5['extra_sparse']['b'])} ({f2(s5['extra_sparse']['se'])}), "
          f"p {pp(s5['extra_sparse']['p'])}; slope at the 90th percentile {f2(s5['extra_sparse']['slope90'][0])}, at the 10th "
          f"{f2(s5['extra_sparse']['slope10'][0])}; Johnson-Neyman share {pc(s5['extra_sparse']['jn_share'])}.",
          f"- **BIC.** Controls {f1(s4['bic']['controls'])}, + six cues {f1(s4['bic']['single'])}, + joint term {f1(s4['bic']['joint'])} "
          "(lower is better; the manuscript reports AIC only).",
          f"- **Random half A.** Passion is significantly negative in "
          f"{pc([x for x in s5['table5'] if x['label'] == 'Random half A'][0]['jn_negative_share'])} of pitches."]

    L += ["", "## 3. Table 5 as computed", "", "| Check | Interaction b (SE), p | Slope at 90th | Slope at 10th | JN share | N (episodes) |", "|---|---|---|---|---|---|"]
    for x in s5["table5"] + [s5["extra_sparse"]]:
        L.append(f"| {x['label']} | {f2(x['b'])} ({f2(x['se'])}), p {pp(x['p'])} | {f2(x['slope90'][0])} | {f2(x['slope10'][0])} | "
                 f"{pc(x['jn_share'])} | {x['n']} ({x['episodes']}) |")
    L += ["", "Figures: " + ", ".join(f"`{p}`" for p in R["step6"]["figures"]), ""]
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")
    return bad, ok


def main():
    quick = "--quick" in sys.argv
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    df = load()
    R = {"meta": dict(quick=quick, pitch_success_rate=float(df.pitch_success.mean()))}
    for name, mod in [("step1", step1_reconciliation), ("step2", step2_boundary), ("step3", step3_pruning),
                      ("step4", step4_ladder), ("step5", step5_robustness)]:
        t = time.time()
        R[name] = mod.run(df, quick=quick)
        print(f"{name} done in {time.time() - t:.0f} s")
    R["step6"] = step6_figures.run(df, quick=quick, curve=R["step2"]["extra_gain_curve"])
    R["examples"] = read_examples.run(df)
    comp = np.array(R["step5"]["r_neg_components"])
    off = comp[np.triu_indices(3, 1)]
    R["step5"]["_r_neg_range"] = f"{r2(off.min())} to {r2(off.max())}"
    R["meta"]["seconds"] = time.time() - t0
    checks = manuscript_checks(R)
    tables = compare_tables(R)
    bad, ok = write_markdown(R, checks, tables, quick)
    (OUT / "results.json").write_text(json.dumps(R, indent=1, default=_jsonable), encoding="utf-8")
    print(f"\n{len(ok) - len(bad)} of {len(ok)} manuscript numbers reproduced; see output/results.md")
    for c in bad:
        print(f"  MISMATCH {c['where']}: {c['what']}: manuscript {c['reported']}, this run {c['run']}")
    print("Largest difference to the unrounded manuscript tables:", f"{max(max(t[2], t[3], t[4]) for t in tables):.1e}")


if __name__ == "__main__":
    main()
