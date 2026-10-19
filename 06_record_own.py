"""Build data/metadata.csv from the transcript table.

Reads data/my_data/transcripts.tsv (filename, Teochew orthography, Mandarin
gloss), locates the corresponding audio under data/my_wav/, and writes
data/metadata.csv for use by 02_transcribe.py and 03_score.py.

Rows beginning with '#' are ignored, so unrecorded items can be left in place.
"""

import sys
from pathlib import Path

WAVDIR = Path("data/my_wav")
TSV = Path("data/my_data/transcripts.tsv")
OUT = Path("data/metadata.csv")


def main():
    if not TSV.exists():
        TSV.parent.mkdir(parents=True, exist_ok=True)
        TSV.write_text(
            "# filename <Tab> Teochew orthography <Tab> Mandarin gloss [<Tab> Peng'im, optional]\n"
            "# Lines beginning with '#' are ignored. Peng'im may be left blank.\n"
            "t01.wav\t我食饭矣\t我吃饭了\n",
            encoding="utf-8",
        )
        print(f"Template created: {TSV}")
        print("Fill in the template and re-run this script.")
        return

    rows, problems = [], []
    with TSV.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [p.strip() for p in line.split("\t")]
            if len(parts) < 3:
                problems.append(f"line {lineno}: fewer than 3 columns (filename<TAB>Teochew<TAB>gloss)")
                continue
            fname, ref_char, ref_gloss = parts[0], parts[1], parts[2]
            ref_pinyin = parts[3] if len(parts) > 3 else ""
            wav = WAVDIR / fname
            if not wav.exists():
                # Also search subdirectories of data/my_wav/
                found = list(WAVDIR.rglob(fname))
                wav = found[0] if found else wav
            if not wav.exists():
                problems.append(f"line {lineno}: audio file not found: {wav}")
                continue
            if ref_char == ref_gloss:
                print(f"  note, line {lineno}: both reference columns are identical ({ref_char}).")
                print("        This is fine if the utterance is identical in both varieties;")
                print("        otherwise the Mandarin gloss column may be blank.")
            rows.append({
                "id": f"own_{lineno:03d}",
                "wav": str(wav),
                "orig_sr": "",
                "ref_char": ref_char,
                "ref_gloss": ref_gloss,
                "ref_pinyin": ref_pinyin,
            })

    if problems:
        print("!! Problems found; fix and re-run:")
        for p in problems:
            print("   -", p)
        return

    if len(rows) < 10:
        print(f"!! Only {len(rows)} utterances; at least 20 is recommended.")
        print("   Fewer than 10 will run but should be interpreted cautiously.")

    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8")
    print(f"OK: wrote {OUT} ({len(rows)} utterances)")
    print()
    print("Spot-check the first three rows:")
    for r in rows[:3]:
        print(f"   {r['id']}  Teochew: {r['ref_char']}   |  Mandarin: {r['ref_gloss']}")
    print()
    print("Next:")
    print("   python 02_transcribe.py --tag A_whisper_large --limit 0")
    print("   python 03_score.py")
    print("   python 04_tone_matrix.py   (optional; requires the Peng'im column)")


if __name__ == "__main__":
    main()
