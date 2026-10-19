"""Step 3 -- score hypotheses against two character-level references.

Computes character error rate (CER) for every utterance against the Teochew
orthographic reference (ref_char) and against the Mandarin gloss (ref_gloss),
after normalisation. Writes per-utterance scores to results/<tag>_scored.csv
and a summary table to results/summary.csv.

CER is not bounded above at 100%: a hypothesis longer than its reference
yields a rate greater than 1.
"""

import re
import glob
from pathlib import Path
import pandas as pd
from jiwer import cer

_SPACE_RE = re.compile(r"[ \t\r\n　]+")
_PUNCT_RE = re.compile(r"[　-〿！-･,.!?;:'\"()\[\]{}<>\-—…·~、，。！？；：（）《》【】]")


def normalize_text(s: str) -> str:
    """Put references and hypotheses on a comparable footing before scoring.

    1. lowercase
    2. remove punctuation (punctuation errors are not phonological errors)
    3. fold whitespace runs to a single space
    4. strip
"""
    if s is None:
        return ""
    s = str(s).lower().strip()
    s = _PUNCT_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s)
    return s.strip()


def cer_of(ref: str, hyp: str) -> float:
    """Compute one CER. Returns nan when the reference is empty (excluded from means).
"""
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    if not ref_n:
        return float("nan")
    return cer(ref_n, hyp_n)


def main():
    # Score model-output files only; exclude summary.csv and *_scored.csv
    files = [f for f in sorted(glob.glob("results/*.csv"))
             if not f.endswith("_scored.csv") and Path(f).name != "summary.csv"]
    if not files:
        print("!! No results/*.csv found; run 02_transcribe.py first")
        return

    summary = []
    for f in files:
        df = pd.read_csv(f)
        if "hyp" not in df.columns:
            continue
        df = df.dropna(subset=["hyp"])

        # Accept either column name: ref_gloss (current) or ref_mnd (legacy)
        gloss_col = "ref_gloss" if "ref_gloss" in df.columns else "ref_mnd"

        df["cer_char"] = [cer_of(a, b) for a, b in zip(df["ref_char"], df["hyp"])]
        df["cer_gloss"] = [cer_of(a, b) for a, b in zip(df[gloss_col], df["hyp"])]

        Path("results").mkdir(exist_ok=True)
        df.to_csv(f.replace(".csv", "_scored.csv"), index=False, encoding="utf-8")

        m_char = df["cer_char"].mean()
        m_gloss = df["cer_gloss"].mean()

        summary.append({
            "model": Path(f).stem,
            "n": len(df),
            "CER_char_%": round(m_char * 100, 2),
            "CER_gloss_%": round(m_gloss * 100, 2),
            "gap_pp": round((m_char - m_gloss) * 100, 2),
        })

    out = pd.DataFrame(summary)
    print()
    print("=========== Dual-reference CER summary ===========")
    print(out.to_string(index=False))
    print()
    print("CER_char   = against the Teochew orthographic reference")
    print("CER_gloss  = against the Mandarin gloss")
    print("gap_pp     = CER_char − CER_gloss")
    print()
    print("Reading the table:")
    print("  gap_pp > 0  →  scoring is more permissive against the gloss")
    print("  gap_pp ~ 0  →  the two references are equivalent; report as a null result")
    print("  gap_pp < 0  →  the orthography is more permissive; investigate")
    out.to_csv("results/summary.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
