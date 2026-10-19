"""Step 2 -- decode reference audio with a Whisper model.

Reads data/metadata.csv, transcribes each listed audio file, and writes one
row per utterance to results/<tag>.csv containing the hypothesis and both
references. The run is resumable: utterances already present in the output
file are skipped, and results are flushed to disk every 10 items.

The language token is forced to "zh": Whisper provides no Teochew token, so
this is the only Sinitic option available under zero-shot decoding.

Usage:
  python 02_transcribe.py --model large-v3 --tag A_whisper_large
  python 02_transcribe.py --model large-v3 --tag A_whisper_large --limit 20
"""

import argparse
import time
from pathlib import Path
import pandas as pd
from faster_whisper import WhisperModel


def check_audio_backend():
    """Verify that an audio decoding backend (av or soundfile) is available.
"""
    ok = []
    for mod in ("av", "soundfile"):
        try:
            __import__(mod); ok.append(mod)
        except Exception:
            pass
    if not ok:
        print("!! No audio decoding backend (av / soundfile) available.")
        print("   Install one first:  pip install -q soundfile")
        print("   Otherwise every utterance fails and hyp is empty.")
        return False
    print(f"Audio decoding backend: {', '.join(ok)}")
    return True


def build_model(model_size, device, compute_type):
    """Load the model. Weights are downloaded on first use and cached thereafter.
"""
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def transcribe_one(model, wav_path, language="zh"):
    """Transcribe a single file and return the concatenated hypothesis text.

    language="zh"  forces Mandarin decoding; Whisper offers no Teochew token.
    beam_size=5    beam width for the decoder.
    vad_filter     removes leading and trailing silence.
"""
    segments, info = model.transcribe(
        str(wav_path),
        language=language,
        beam_size=5,
        vad_filter=True,
    )
    return "".join(seg.text for seg in segments).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="medium",
                    help="faster-whisper model size (large-v3, medium, small)")
    ap.add_argument("--tag", required=True,
                    help="tag for the output file, e.g. A_whisper_large")
    ap.add_argument("--device", default="auto", help="auto / cuda / cpu")
    ap.add_argument("--compute", default="default",
                    help="default / float16 / int8 (float16 on GPU, int8 on CPU)")
    ap.add_argument("--limit", type=int, default=0, help="0 = all utterances")
    ap.add_argument("--language", default="zh")
    args = ap.parse_args()

    meta = pd.read_csv("data/metadata.csv")
    if args.limit:
        meta = meta.head(args.limit)

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{args.tag}.csv"

    # Resume support: skip ids already present in the output
    done = set()
    if out_path.exists():
        done = set(pd.read_csv(out_path)["id"].astype(str))
        print(f"{len(done)} utterances already decoded, skipping.")

    print(f"Loading model {args.model} ... (first run downloads weights)")
    if args.device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    else:
        device = args.device
    compute = "float16" if (device == "cuda" and args.compute == "default") else ("int8" if device == "cpu" and args.compute == "default" else args.compute)
    print(f"device={device}  compute_type={compute}")
    if not check_audio_backend():
        return
    model = build_model(args.model, device, compute)

    rows = []
    _err_shown = [0]
    t0 = time.time()
    for i, r in enumerate(meta.itertuples()):
        if str(r.id) in done:
            continue
        wav_path = Path(str(r.wav))
        if not wav_path.exists():
            found = list(Path("data").rglob(wav_path.name)) if Path("data").exists() else []
            wav_path = found[0] if found else wav_path
        if not wav_path.exists():
            print(f"  !! {r.id}: audio file not found: {r.wav}")
            hyp = ""
        else:
            try:
                hyp = transcribe_one(model, wav_path, args.language)
            except Exception as e:
                # Print the full exception rather than swallowing it
                import traceback
                print(f"  !! {r.id} failed: {type(e).__name__}: {e}")
                if _err_shown[0] < 3:
                    traceback.print_exc()
                    _err_shown[0] += 1
                hyp = ""

        row = {"id": r.id, "hyp": hyp, "ref_char": r.ref_char}
        # Carry reference columns through; 03_score.py needs ref_gloss
        if hasattr(r, "ref_gloss"):
            row["ref_gloss"] = r.ref_gloss
        if hasattr(r, "ref_pinyin"):
            row["ref_pinyin"] = r.ref_pinyin
        rows.append(row)

        if (len(rows)) % 10 == 0:
            # Flush to disk every 10 items
            new = pd.DataFrame(rows)
            if out_path.exists():
                new = pd.concat([pd.read_csv(out_path), new], ignore_index=True)
            new.to_csv(out_path, index=False, encoding="utf-8")
            rows = []
            el = time.time() - t0
            print(f"  {i + 1}/{len(meta)}  elapsed {el / 60:.1f} min")

    if rows:
        new = pd.DataFrame(rows)
        if out_path.exists():
            new = pd.concat([pd.read_csv(out_path), new], ignore_index=True)
        new.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Done: {out_path} ({len(pd.read_csv(out_path))} utterances)")


if __name__ == "__main__":
    main()
