#!/usr/bin/env python3
"""Verbal features for the pitch corpus, on the timestamps of the transcript.

Reads the transcript segments (data/inputs/mediaflow/per_frame/voice_asr.csv) and writes
  verbal_features_segments.csv   one row per transcript segment (same timestamps as voice_asr.csv)
  verbal_features_pitch.csv      one row per pitch (rates per 100 words, means, readability)
  verbal_features_columns.csv    column dictionary: family, description, tool, licence

Everything runs on free software. spaCy, VADER, and Empath need no extra files. The Loughran-McDonald
dictionary and the two NRC lexicons are NOT part of this repository; download them yourself and put them
into the lexicons folder (see lexicons/README.md). A family whose lexicon is missing is skipped and
reported; all other families are still written. The NRC families are optional.

    python verbal/build_verbal_features.py --replication . --lexicons verbal/lexicons --out verbal/out

Options: --empath subset|all|none (default: 20 pitch-relevant Empath topics).

Counting rules (identical to the rules behind the word measures in the manuscript):
  * words are all tokens that are neither punctuation nor white space (spaCy tokenizer, en_core_web_sm);
  * Loughran-McDonald and NRC words are matched on the word form and, if that fails, on the lemma;
    Empath and regulatory-focus words count if the word form or the lemma is on the list;
  * segment-level columns are raw counts (so any slice of a pitch can be rebuilt by summing segments);
    pitch-level columns are counts per 100 words unless stated otherwise.
"""
import argparse, glob, json, os, re, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--replication", required=True, help="folder that contains data/ (this package, if you run it from the package root)")
ap.add_argument("--lexicons", default="lexicons", help="folder with the lexicons you downloaded")
ap.add_argument("--out", default="out")
ap.add_argument("--empath", choices=["subset", "all", "none"], default="subset", help="Empath topics: 20 pitch-relevant categories (default), all 194, or none")
ap.add_argument("--wordlists", default=None, help="folder with regulatory_focus_gamache2015.txt (default: wordlists/ next to this script)")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
DATA = os.path.join(a.replication, "data"); PF = os.path.join(DATA, "inputs", "mediaflow", "per_frame")

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nlp = spacy.load("en_core_web_sm"); vader = SentimentIntensityAnalyzer()
COLS = []                                   # column dictionary: (column, family, description, tool, licence)
def doc_col(col, fam, desc, tool, lic): COLS.append(dict(column=col, family=fam, description=desc, tool=tool, licence=lic))

# ------------------------------------------------------------------ lexicons (optional, never bundled)
def first(*patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits: return hits[0]
    return None

LM = None; f = first(os.path.join(a.lexicons, "*MasterDictionary*.csv"), os.path.join(a.lexicons, "LM.csv"))
LM_CATS = ["lm_negative", "lm_positive", "lm_uncertainty", "lm_litigious", "lm_constraining", "lm_strong_modal", "lm_weak_modal"]
if f:
    t = pd.read_csv(f, keep_default_na=False, low_memory=False); t["Word"] = t.Word.astype(str).str.upper(); LM = {}
    num = lambda c: pd.to_numeric(t[c], errors="coerce").fillna(0)
    for c, lab in [("Negative", "lm_negative"), ("Positive", "lm_positive"), ("Uncertainty", "lm_uncertainty"), ("Litigious", "lm_litigious"), ("Constraining", "lm_constraining")]:
        for w in t.loc[num(c) > 0, "Word"]: LM.setdefault(w, set()).add(lab)
    if "Strong_Modal" in t.columns:          # current files
        for w in t.loc[num("Strong_Modal") > 0, "Word"]: LM.setdefault(w, set()).add("lm_strong_modal")
        for w in t.loc[num("Weak_Modal") > 0, "Word"]: LM.setdefault(w, set()).add("lm_weak_modal")
    elif "Modal" in t.columns:               # older files: 1 = strong, 3 = weak
        for w in t.loc[num("Modal") == 1, "Word"]: LM.setdefault(w, set()).add("lm_strong_modal")
        for w in t.loc[num("Modal") == 3, "Word"]: LM.setdefault(w, set()).add("lm_weak_modal")
    print("Loughran-McDonald:", os.path.basename(f), len(LM), "words")
else: print("SKIPPED Loughran-McDonald: no *MasterDictionary*.csv or LM.csv in", a.lexicons)

EMO = None; EMO_CATS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive", "negative"]
f = first(os.path.join(a.lexicons, "nrc_emolex.json"), os.path.join(a.lexicons, "NRC-Emotion-Lexicon-Wordlevel*.txt"))
if f and f.endswith(".json"): EMO = {k.lower(): set(v) for k, v in json.load(open(f)).items()}
elif f:
    EMO = {}
    for line in open(f, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 3 and p[2].strip() == "1": EMO.setdefault(p[0].lower(), set()).add(p[1])
if EMO is not None: print("NRC Emotion Lexicon:", os.path.basename(f), len(EMO), "words")
else: print("SKIPPED NRC Emotion Lexicon: no nrc_emolex.json or NRC-Emotion-Lexicon-Wordlevel*.txt in", a.lexicons)

VAD = None; f = first(os.path.join(a.lexicons, "NRC-VAD-Lexicon*.txt"))
if f:
    t = pd.read_csv(f, sep="\t", keep_default_na=False); t.columns = [c.lower() for c in t.columns]; w = t.columns[0]
    VAD = {str(k).lower(): (v, r, d) for k, v, r, d in zip(t[w], t["valence"].astype(float), t["arousal"].astype(float), t["dominance"].astype(float))}
    print("NRC-VAD:", os.path.basename(f), len(VAD), "words")
else: print("SKIPPED NRC-VAD: no NRC-VAD-Lexicon*.txt in", a.lexicons)

EMPATH_SUBSET = ["business", "economics", "money", "payment", "banking", "valuable", "gain", "work", "achievement", "leader", "power", "competing",
                 "negotiate", "technology", "science", "help", "trust", "optimism", "positive_emotion", "negative_emotion"]
EMP = None
if a.empath != "none":
    try:
        from empath import Empath
        cats = Empath().cats
        if a.empath == "subset": cats = {c: cats[c] for c in EMPATH_SUBSET}
        EMP = {}; EMP_CATS = sorted(cats)
        for c, ws in cats.items():
            for w in ws: EMP.setdefault(w.lower(), set()).add(c)
        print("Empath:", len(EMP_CATS), "categories")
    except ImportError: print("SKIPPED Empath: pip install empath")

RF = None; f = os.path.join(a.wordlists or os.path.join(os.path.dirname(os.path.abspath(__file__)), "wordlists"), "regulatory_focus_gamache2015.txt")
if os.path.exists(f):
    RF = {}; cur = None
    for line in open(f, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if line.startswith("["): cur = line.strip("[]"); continue
        for w in line.split(): RF.setdefault(w.lower(), set()).add(cur)
    print("Regulatory focus:", len(RF), "words")
else: print("SKIPPED regulatory focus: no", f)

# ------------------------------------------------------------------ grammar (no dictionary needed)
I_ = set("i me my mine myself".split()); WE = set("we us our ours ourselves".split()); YOU = set("you your yours yourself yourselves".split())
SHEHE = set("he him his himself she her hers herself".split()); THEY = set("they them their theirs themselves".split())
IPRON = set("it its itself this that these those something anything everything nothing someone anyone everyone nobody somebody anybody everybody what which who whom whose whatever whichever one ones another other others".split())
NEG = set("no not never n't none nothing nobody neither nor nowhere without cannot".split())
GRAMMAR = [("g_article", "articles (a, an, the)"), ("g_preposition", "prepositions (part-of-speech tag ADP)"),
    ("g_pron_i", "first person singular pronouns"), ("g_pron_we", "first person plural pronouns"), ("g_pron_you", "second person pronouns"),
    ("g_pron_shehe", "third person singular pronouns"), ("g_pron_they", "third person plural pronouns"), ("g_pron_impersonal", "impersonal pronouns (it, this, something ...)"),
    ("g_auxiliary", "auxiliary verbs (tag AUX)"), ("g_conjunction", "conjunctions (tags CCONJ, SCONJ)"), ("g_adverb", "adverbs (tag ADV, without negations)"),
    ("g_negation", "negations (no, not, never ...)"), ("g_verb", "main verbs (tag VERB)"), ("g_adjective", "adjectives (tag ADJ)"), ("g_noun", "common nouns (tag NOUN)"),
    ("g_proper_noun", "proper nouns (tag PROPN)"), ("g_number", "numbers (tag NUM)"), ("g_past_tense", "verbs in past tense"), ("g_present_tense", "verbs in present tense"),
    ("g_future", "future markers (will, shall, 'll, gonna, going to)"), ("g_modal", "modal verbs (tag MD)"), ("g_comparative", "comparatives and superlatives"),
    ("g_wh_word", "question words (who, what, why, how ...)")]
ENTS = [("ent_money", ("MONEY",), "money amounts"), ("ent_percent", ("PERCENT",), "percentages"), ("ent_quantity", ("CARDINAL", "QUANTITY"), "other counts and quantities"),
    ("ent_time", ("DATE", "TIME"), "dates and times"), ("ent_organization", ("ORG",), "organizations"), ("ent_place", ("GPE", "LOC"), "places"), ("ent_person", ("PERSON",), "persons")]

def grammar_counts(words):
    c = dict.fromkeys([g for g, _ in GRAMMAR], 0)
    for i, t in enumerate(words):
        lo, po = t.lower_, t.pos_
        if lo in ("a", "an", "the"): c["g_article"] += 1
        elif po == "ADP": c["g_preposition"] += 1
        if lo in I_: c["g_pron_i"] += 1
        elif lo in WE: c["g_pron_we"] += 1
        elif lo in YOU: c["g_pron_you"] += 1
        elif lo in SHEHE: c["g_pron_shehe"] += 1
        elif lo in THEY: c["g_pron_they"] += 1
        elif lo in IPRON and po in ("PRON", "DET"): c["g_pron_impersonal"] += 1
        if po == "AUX": c["g_auxiliary"] += 1
        if po in ("CCONJ", "SCONJ"): c["g_conjunction"] += 1
        if po == "ADV" and lo not in NEG: c["g_adverb"] += 1
        if lo in NEG: c["g_negation"] += 1
        if po == "VERB": c["g_verb"] += 1
        if po == "ADJ": c["g_adjective"] += 1
        if po == "NOUN": c["g_noun"] += 1
        if po == "PROPN": c["g_proper_noun"] += 1
        if po == "NUM": c["g_number"] += 1
        tense = t.morph.get("Tense")
        if po in ("VERB", "AUX") and "Past" in tense: c["g_past_tense"] += 1
        if po in ("VERB", "AUX") and "Pres" in tense: c["g_present_tense"] += 1
        nxt = words[i + 1].lower_ if i + 1 < len(words) else ""
        if (t.tag_ == "MD" and lo in ("will", "shall", "'ll", "wo")) or lo == "gon" or (lo == "going" and nxt == "to"): c["g_future"] += 1
        if t.tag_ == "MD": c["g_modal"] += 1
        if t.tag_ in ("JJR", "JJS", "RBR", "RBS"): c["g_comparative"] += 1
        if t.tag_ in ("WDT", "WP", "WP$", "WRB"): c["g_wh_word"] += 1
    return c

def cdi(c, n):
    """Categorical-dynamic index (Pennebaker et al., 2014), in percent of words; an open counterpart of LIWC's Analytic score."""
    if n == 0: return np.nan
    p = lambda k: 100 * c[k] / n
    ppron = sum(p(k) for k in ("g_pron_i", "g_pron_we", "g_pron_you", "g_pron_shehe", "g_pron_they"))
    return 30 + p("g_article") + p("g_preposition") - ppron - p("g_pron_impersonal") - p("g_auxiliary") - p("g_conjunction") - p("g_adverb") - p("g_negation")

# ------------------------------------------------------------------ read transcript segments
asr = pd.read_csv(os.path.join(PF, "voice_asr.csv")); asr["subsegment_id"] = asr.video_id.str.split("/").str[-1]
asr["text"] = asr.text.fillna("").astype(str)
try:
    af = pd.read_csv(os.path.join(DATA, "presenter.csv"))
    pres = dict(zip(af.subsegment_id, af.presenter_person_id.astype(str)))
    asr = asr[asr.subsegment_id.isin(pres)].copy()
    asr["is_presenter"] = (asr.person_id.astype(str) == asr.subsegment_id.map(pres)).astype(int)
except Exception as e:
    print("presenter.csv not read (", type(e).__name__, "): keeping all videos, no presenter flag"); asr["is_presenter"] = np.nan
asr = asr.sort_values(["subsegment_id", "t_start_s"]).reset_index(drop=True)
asr["pause_before_s"] = (asr.t_start_s - asr.groupby("subsegment_id").t_end_s.shift()).clip(lower=0)

rows = []; pitch_tokens = {}
for (sid, dur), doc in zip(zip(asr.subsegment_id, asr.duration_s), nlp.pipe(asr.text, batch_size=64)):
    toks = [t for t in doc if not t.is_space]; words = [t for t in toks if not t.is_punct]; n = len(words)
    r = dict(n_words=n, n_sentences=len(list(doc.sents)) if n else 0, n_letters=sum(len(t.text) for t in words),
             n_long_words=sum(len(t.text) >= 7 for t in words), n_question_marks=sum(t.text == "?" for t in toks),
             words_per_second=n / dur if dur and dur > 0 else np.nan)
    g = grammar_counts(words); r.update(g)
    for col, labels, _ in ENTS: r[col] = sum(e.label_ in labels for e in doc.ents)
    ps = vader.polarity_scores(doc.text) if doc.text.strip() else dict(neg=np.nan, neu=np.nan, pos=np.nan, compound=np.nan)
    r.update(vader_negative=ps["neg"], vader_neutral=ps["neu"], vader_positive=ps["pos"], vader_compound=ps["compound"])
    look = lambda d, t: d.get(t.lower_, d.get(t.lemma_.lower()))                      # word form, else lemma (rule of the manuscript)
    both = lambda d, t: d.get(t.lower_, set()) | d.get(t.lemma_.lower(), set())      # word form or lemma (Empath, regulatory focus)
    if LM is not None:
        hit = [LM.get(t.text.upper(), LM.get(t.lemma_.upper(), ())) for t in words]
        for c in LM_CATS: r[c] = sum(c in h for h in hit)
    if EMO is not None:
        hit = [look(EMO, t) or () for t in words]
        for c in EMO_CATS: r["nrc_" + c] = sum(c in h for h in hit)
    if VAD is not None:
        hit = [x for x in (look(VAD, t) for t in words) if x is not None]
        r["nrcvad_n_matched"] = len(hit)
        for j, c in enumerate(("valence", "arousal", "dominance")): r["nrcvad_" + c] = float(np.mean([h[j] for h in hit])) if hit else np.nan
    if EMP is not None:
        hit = [both(EMP, t) for t in words]
        for c in EMP_CATS: r["empath_" + c] = sum(c in h for h in hit)
    if RF is not None:
        hit = [both(RF, t) for t in words]
        for c in ("promotion", "prevention"): r["rf_" + c] = sum(c in h for h in hit)
    rows.append(r); pitch_tokens.setdefault(sid, []).extend(t.lower_ for t in words)
F = pd.DataFrame(rows)
seg = pd.concat([asr[["subsegment_id", "video_id", "t_start_s", "t_end_s", "duration_s", "speaker_id", "person_id", "is_presenter", "pause_before_s"]], F], axis=1)
seg.to_csv(os.path.join(a.out, "verbal_features_segments.csv"), index=False)

# ------------------------------------------------------------------ pitch level
def syllables(w):
    """Heuristic syllable count: vowel groups, silent final e removed, at least one."""
    w = re.sub(r"[^a-z]", "", w.lower())
    if not w: return 0
    k = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee")) and k > 1: k -= 1
    return max(k, 1)
count_cols = [c for c in F.columns if c.startswith(("g_", "ent_", "lm_", "nrc_", "empath_", "rf_")) and not c.startswith("nrcvad")] + ["n_long_words", "n_question_marks"]
def mattr(tokens, w=50):
    if len(tokens) < w: return len(set(tokens)) / len(tokens) if tokens else np.nan
    return float(np.mean([len(set(tokens[i:i + w])) / w for i in range(len(tokens) - w + 1)]))
P = []
for sid, g in seg.groupby("subsegment_id", sort=True):
    n = g.n_words.sum(); text = " ".join(asr.loc[g.index, "text"]); span = g.t_end_s.max() - g.t_start_s.min()
    r = dict(subsegment_id=sid, n_segments=len(g), n_words=int(n), n_sentences=int(g.n_sentences.sum()),
             words_per_minute=60 * n / span if span > 0 else np.nan, pause_share=1 - g.duration_s.sum() / span if span > 0 else np.nan,
             mean_word_length=g.n_letters.sum() / n if n else np.nan, words_per_sentence=n / g.n_sentences.sum() if g.n_sentences.sum() else np.nan,
             type_token_ratio_mattr50=mattr(pitch_tokens.get(sid, [])))
    toks_ = pitch_tokens.get(sid, []); syl = sum(syllables(w) for w in toks_); ns = g.n_sentences.sum()
    r["flesch_reading_ease"] = 206.835 - 1.015 * n / ns - 84.6 * syl / n if n and ns else np.nan
    r["flesch_kincaid_grade"] = 0.39 * n / ns + 11.8 * syl / n - 15.59 if n and ns else np.nan
    for c in count_cols: r[c.replace("n_long_words", "long_words").replace("n_question_marks", "question_marks")] = 100 * g[c].sum() / n if n else np.nan
    r["open_analytic_cdi"] = cdi({k: g[k].sum() for k, _ in GRAMMAR}, n)
    ps = vader.polarity_scores(text) if text.strip() else dict(neg=np.nan, neu=np.nan, pos=np.nan, compound=np.nan)
    r.update(vader_negative=ps["neg"], vader_neutral=ps["neu"], vader_positive=ps["pos"], vader_compound=ps["compound"], vader_compound_segment_mean=g.vader_compound.mean())
    if VAD is not None:
        m = g.nrcvad_n_matched.sum(); r["nrcvad_share_matched"] = m / n if n else np.nan
        for c in ("valence", "arousal", "dominance"): r["nrcvad_" + c] = (g["nrcvad_" + c] * g.nrcvad_n_matched).sum() / m if m else np.nan
    P.append(r)
P = pd.DataFrame(P); P.to_csv(os.path.join(a.out, "verbal_features_pitch.csv"), index=False)

# ------------------------------------------------------------------ column dictionary
SP = ("spaCy 3 with en_core_web_sm", "MIT"); 
for c, d in [("n_words", "words (tokens that are neither punctuation nor white space)"), ("n_sentences", "sentences (spaCy sentence boundaries)"), ("n_letters", "characters in words (segment file)"),
             ("n_long_words / long_words", "words of seven or more characters"), ("n_question_marks / question_marks", "question marks"), ("words_per_second", "words per second of the segment (segment file)"),
             ("pause_before_s", "silence between the previous segment and this one, in seconds (segment file)"), ("words_per_minute", "words per minute over the whole pitch (pitch file)"),
             ("pause_share", "share of the pitch without transcribed speech (pitch file)"), ("mean_word_length", "characters per word (pitch file)"), ("words_per_sentence", "words per sentence (pitch file)"),
             ("type_token_ratio_mattr50", "lexical diversity: moving-average type-token ratio, window of 50 words (pitch file)")]: doc_col(c, "A fluency and complexity", d, *SP)
for c, d in [("flesch_reading_ease", "Flesch reading ease, with a heuristic syllable count (pitch file)"), ("flesch_kincaid_grade", "Flesch-Kincaid grade level, with a heuristic syllable count (pitch file)")]: doc_col(c, "A fluency and complexity", d, "formulas of Flesch (1948) and Kincaid et al. (1975), computed in this script", "MIT (code)")
for c, d in GRAMMAR: doc_col(c, "B grammar and function words", d, *SP)
doc_col("open_analytic_cdi", "B grammar and function words", "categorical-dynamic index: articles + prepositions - pronouns - auxiliary verbs - conjunctions - adverbs - negations, + 30 (pitch file); open counterpart of LIWC Analytic", "formula of Pennebaker et al. (2014), computed with spaCy", "MIT (code); formula published open access")
for c, _, d in ENTS: doc_col(c, "C figures and named entities", d + " (spaCy named entities)", *SP)
for c, d in [("vader_negative", "share of negative sentiment"), ("vader_neutral", "share of neutral sentiment"), ("vader_positive", "share of positive sentiment"), ("vader_compound", "compound sentiment, -1 to 1"), ("vader_compound_segment_mean", "mean compound sentiment of the segments (pitch file)")]:
    doc_col(c, "D sentiment (VADER)", d, "vaderSentiment (Hutto & Gilbert, 2014)", "MIT; lexicon ships with the package")
if LM is not None:
    for c in LM_CATS: doc_col(c, "E business tone (Loughran-McDonald)", c[3:].replace("_", " ") + " words", "Loughran-McDonald Master Dictionary (Loughran & McDonald, 2011)", "free for academic research; not bundled")
if EMO is not None:
    for c in EMO_CATS: doc_col("nrc_" + c, "F emotion words (NRC)", c + " words", "NRC Emotion Lexicon (Mohammad & Turney, 2013)", "free for research; not bundled")
if VAD is not None:
    for c in ("valence", "arousal", "dominance"): doc_col("nrcvad_" + c, "G valence, arousal, dominance of words (NRC-VAD)", "mean " + c + " of the matched words", "NRC-VAD Lexicon (Mohammad, 2018)", "free for non-commercial research; do not redistribute; not bundled")
    doc_col("nrcvad_n_matched / nrcvad_share_matched", "G valence, arousal, dominance of words (NRC-VAD)", "words found in the lexicon (count in the segment file, share in the pitch file)", "NRC-VAD Lexicon (Mohammad, 2018)", "see above")
if EMP is not None:
    for c in EMP_CATS: doc_col("empath_" + c, "H topics (Empath)", "words of the Empath category '" + c + "'", "Empath (Fast, Chen, & Bernstein, 2016)", "MIT; word lists ship with the package")
if RF is not None:
    for c in ("promotion", "prevention"): doc_col("rf_" + c, "I regulatory focus", c + "-focus words (27 promotion words, 25 prevention words)", "word lists of Gamache, McNamara, Mannor, & Johnson (2015)", "printed in the article; no reuse terms stated")
C = pd.DataFrame(COLS); C.to_csv(os.path.join(a.out, "verbal_features_columns.csv"), index=False)
print("\nsegments:", seg.shape, "| pitches:", P.shape); print(C.groupby("family").size().to_string())
