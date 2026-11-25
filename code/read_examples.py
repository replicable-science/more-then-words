"""Face validity of the moderator: the transcripts of the pitches with the most and the fewest
negative words, with facial energy and the funding decision. Writes output/examples_negative_words.md.

    python code/read_examples.py        (default: 8 pitches at each end)
    python code/read_examples.py 15
"""
import sys

import pandas as pd

from common import DATA, OUT, load


def run(df=None, k=8):
    df = load() if df is None else df
    tx = pd.read_csv(DATA / "transcripts.csv").set_index("subsegment_id").full_text
    d = df.sort_values("neg_words")
    L = ["# Pitches with the most and the fewest negative words", "",
         "Negative words and facial energy are z-scores over the 457 pitches. Funded: the panel's decision.", ""]
    for title, part in [(f"## {k} pitches with the MOST negative words", d.tail(k).iloc[::-1]),
                        (f"## {k} pitches with the FEWEST negative words", d.head(k))]:
        L += [title, ""]
        for _, r in part.iterrows():
            L += [f"**{r.subsegment_id}**: negative words {r.neg_words:+.2f}, facial energy {r.face_energy:+.2f}, "
                  f"funded {'yes' if r.funding_success == 1 else 'no'}", "", "> " + str(tx.get(r.subsegment_id, "")).strip(), ""]
    OUT.mkdir(exist_ok=True)
    (OUT / "examples_negative_words.md").write_text("\n".join(L), encoding="utf-8")
    return {"examples": "output/examples_negative_words.md"}


if __name__ == "__main__":
    print(run(k=int(sys.argv[1]) if len(sys.argv) > 1 else 8))
