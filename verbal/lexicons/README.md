# Lexicons that are not part of this repository

Three lexicons are free for research but may not be redistributed (or their terms do not say). Download them yourself and put the files into this folder. The script finds them by file name and skips a family whose file is missing. Loughran-McDonald belongs to the standard set; the two NRC lexicons are optional.

| Lexicon | Where to get it | File the script looks for |
|---|---|---|
| Loughran-McDonald Master Dictionary | https://sraf.nd.edu/loughranmcdonald-master-dictionary/ | `*MasterDictionary*.csv` (or `LM.csv`) |
| NRC Emotion Lexicon | https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm | `NRC-Emotion-Lexicon-Wordlevel*.txt` (or `nrc_emolex.json`) |
| NRC-VAD Lexicon | https://www.saifmohammad.com/WebPages/nrc-vad.html | `NRC-VAD-Lexicon*.txt` |

VADER and Empath need no download here: their word lists ship with the Python packages (both MIT). This script does not include a regulatory-focus word list.

The manuscript used version 1 of NRC-VAD (2018, scores from 0 to 1). Version 2.1 (2025) has more words; if you switch, the verbal energy measure changes and must be re-estimated.
