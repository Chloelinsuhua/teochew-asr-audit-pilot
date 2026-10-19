"""Convert recordings to 16 kHz mono WAV and normalise filenames.

Phone voice recorders save m4a (AAC); faster-whisper and soundfile expect
WAV. Decoding uses PyAV (pip install av), which bundles its own ffmpeg
libraries, so no system ffmpeg installation is required.

The script:
  1. finds all audio files under data/my_wav/, including double extensions
     such as .wav.m4a and misspellings such as .waw.m4a;
  2. decodes, resamples to 16000 Hz, downmixes to mono, and writes .wav;
  3. normalises filenames (sp01_teo.wav.m4a -> sp01_teo.wav).

Usage:
  python 07_convert_audio.py
"""

import re
from pathlib import Path
import numpy as np
import soundfile as sf

SRC = Path("data/my_wav")
TARGET_SR = 16000


def decode_m4a(path):
    """Decode an audio file with PyAV and return (mono float32 waveform, sample rate).
"""
    import av
    container = av.open(str(path))
    stream = container.streams.audio[0]
    sr = stream.rate
    chunks = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
    for frame in container.decode(stream):
        for out in resampler.resample(frame):
            arr = out.to_ndarray()
            chunks.append(arr.reshape(-1))
    if not chunks:
        return None, None
    wav = np.concatenate(chunks).astype(np.float32)
    return wav, sr


def clean_name(path: Path) -> str:
    """Normalise file extensions to .wav.
"""
    name = path.name
    # strip the trailing .m4a
    name = re.sub(r"\.m4a$", "", name, flags=re.I)
    # strip the spurious intermediate .wav / .waw
    name = re.sub(r"\.wa[vw]$", "", name, flags=re.I)
    if not name.lower().endswith(".wav"):
        name += ".wav"
    return name


def main():
    files = sorted([p for p in SRC.iterdir()
                    if p.suffix.lower() == ".m4a" and not p.name.startswith(".")])
    if not files:
        print("No m4a files found (possibly already converted).")
        return

    print(f"Found {len(files)} m4a files; converting...\n")
    ok = fail = 0
    for p in files:
        try:
            wav, sr = decode_m4a(p)
            if wav is None:
                print(f"  !! {p.name}: decoded waveform is empty, skipping")
                fail += 1
                continue

            # resample to 16000 Hz (the rate Whisper expects)
            if sr != TARGET_SR:
                x_old = np.arange(len(wav), dtype=np.float32) / sr
                n_new = int(round(len(wav) * TARGET_SR / sr))
                x_new = np.arange(n_new, dtype=np.float32) / TARGET_SR
                wav = np.interp(x_new, x_old, wav).astype(np.float32)

            out = SRC / clean_name(p)
            sf.write(str(out), wav, TARGET_SR)
            dur = len(wav) / TARGET_SR
            print(f"  OK {p.name:<24} -> {out.name:<18} {dur:.1f}s  ({sr}Hz -> {TARGET_SR}Hz)")
            ok += 1
        except Exception as e:
            print(f"  !! {p.name} failed: {type(e).__name__}: {str(e)[:100]}")
            fail += 1

    print(f"\nDone: {ok} converted, {fail} failed")
    print("Original m4a files are preserved and may be deleted once verified.")


if __name__ == "__main__":
    main()
