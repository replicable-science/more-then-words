"""Rebuild joy, smile, and positive face from the frame table and check them against data/pitches.csv.

Not needed for `python run_all.py`. The mean is taken over the frames of the presenting founder
(the person linked to the pitch; 95,661 frames in 457 pitches).

    python prep/build_face_measures.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
pres = pd.read_csv(ROOT / "data" / "presenter.csv")
presenter = dict(zip(pres.subsegment_id, pres.presenter_person_id))
b = pd.read_parquet(
    ROOT / "data" / "inputs" / "mediaflow" / "per_frame" / "body_frames.parquet",
    columns=["video_id", "person_id", "emo_happiness", "au_AU6", "au_AU12", "valence", "arousal"],
)
b["sid"] = b.video_id.str.split("/").str[-1]
b = b[b.person_id == b.sid.map(presenter)].copy()
b["smile"] = np.minimum(b.au_AU6, b.au_AU12)
pf = b.groupby("sid").agg(
    face_joy_pf=("emo_happiness", "mean"),
    face_smile_au_pf=("smile", "mean"),
    face_valence_pf=("valence", "mean"),
    n_frames=("valence", "size"),
)
print(f"{len(pf)} pitches, {int(pf.n_frames.sum()):,} presenter frames")
d = pd.read_csv(ROOT / "data" / "pitches.csv").set_index("subsegment_id")
for c in ["face_joy_pf", "face_smile_au_pf", "face_valence_pf"]:
    diff = (pf[c].reindex(d.index) - d[c]).abs().max()
    print(f"{c}: max difference to data/pitches.csv = {diff:.1e}")
