#!/usr/bin/env python3
"""Preprocess Vietnamese dataset (ngochuyen001) for Kokoro-7M Distillation.
1. Reads metadata.csv (filename|text).
2. Trims leading/trailing silence (top_db=35).
3. Resamples to 24000 Hz mono (FLAC format).
4. Converts Vietnamese text to Kokoro 178-token IDs using vig2p + kokoro_symbols.
5. Emits vi_manifest.jsonl: {"wav": ..., "ids": [...], "text": ..., "duration": ...}
"""
import _env  # noqa: F401
import argparse
import json
import os
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from vig2p import VietnameseG2P
from kokoro_symbols import dicts

SR = 24000


def find_audio_file(data_dir: str, fname: str) -> str | None:
    candidates = [
        os.path.join(data_dir, fname),
        os.path.join(data_dir, "wavs", fname),
        os.path.join(data_dir, fname + ".wav"),
        os.path.join(data_dir, "wavs", fname + ".wav"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess Vietnamese dataset for Kokoro-7M")
    parser.add_argument(
        "--data-dir",
        default="/kaggle/input/datasets/xkorenguyenthinh/ngochuyen001/content/audio",
        help="Root directory containing audio and metadata.csv",
    )
    parser.add_argument(
        "--metadata",
        default=None,
        help="Path to metadata.csv (defaults to <data-dir>/metadata.csv)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/wavs",
        help="Directory to save preprocessed 24kHz FLAC audio files",
    )
    parser.add_argument(
        "--out-manifest",
        default="data/vi_manifest.jsonl",
        help="Output manifest JSONL path",
    )
    parser.add_argument("--sr", type=int, default=SR, help="Target sample rate (default: 24000)")
    parser.add_argument("--top-db", type=float, default=35.0, help="Trim silence top_db (default: 35.0)")
    parser.add_argument("--min-dur", type=float, default=0.5, help="Min duration in seconds")
    parser.add_argument("--max-dur", type=float, default=30.0, help="Max duration in seconds")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 for all)")

    args = parser.parse_args()

    meta_file = args.metadata or os.path.join(args.data_dir, "metadata.csv")
    if not os.path.exists(meta_file):
        # Check alternate location
        alt_meta = os.path.join(args.data_dir, "metadata.txt")
        if os.path.exists(alt_meta):
            meta_file = alt_meta
        else:
            print(f"Error: metadata file not found at {meta_file}", file=sys.stderr)
            sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_manifest)), exist_ok=True)

    print(f"[*] Loading Vietnamese G2P...")
    g2p = VietnameseG2P()
    vocab = dicts

    kept = 0
    dropped = 0
    total_sec = 0.0
    trimmed_sec = 0.0
    t0 = time.time()

    print(f"[*] Reading metadata from: {meta_file}")
    print(f"[*] Writing audio to: {args.out_dir}")
    print(f"[*] Writing manifest to: {args.out_manifest}")

    with open(meta_file, "r", encoding="utf-8") as f, open(args.out_manifest, "w", encoding="utf-8") as out_f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            delimiter = "|" if "|" in line else "\t" if "\t" in line else ","
            parts = line.split(delimiter, 1)
            if len(parts) < 2:
                dropped += 1
                continue

            fname, text = parts[0].strip(), parts[1].strip()
            # If metadata has third column (e.g. speaker), strip it out from text if needed
            if "|" in text:
                text = text.split("|")[0].strip()

            audio_path = find_audio_file(args.data_dir, fname)
            if not audio_path:
                dropped += 1
                continue

            try:
                y, sr = sf.read(audio_path, dtype="float32")
            except Exception as e:
                dropped += 1
                continue

            if y.ndim > 1:
                y = y.mean(axis=1)

            raw_dur = len(y) / sr
            # Trim conservative silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=args.top_db)
            if sr != args.sr:
                y_trimmed = librosa.resample(y_trimmed, orig_sr=sr, target_sr=args.sr)

            dur = len(y_trimmed) / args.sr
            if not (args.min_dur <= dur <= args.max_dur):
                dropped += 1
                continue

            # G2P conversion
            try:
                phonemes = g2p(text)
                token_ids = [vocab[c] for c in phonemes if c in vocab]
            except Exception:
                dropped += 1
                continue

            if not token_ids or len(token_ids) < 2:
                dropped += 1
                continue

            # Save as 24kHz FLAC
            base_name = Path(fname).stem
            out_audio_path = os.path.join(args.out_dir, f"{base_name}.flac")
            sf.write(out_audio_path, y_trimmed, args.sr, format="FLAC")

            row = {
                "wav": os.path.abspath(out_audio_path),
                "ids": token_ids,
                "text": text,
                "duration": round(dur, 3),
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")

            kept += 1
            total_sec += dur
            trimmed_sec += max(0.0, raw_dur - dur)

            if kept % 1000 == 0:
                print(
                    f"  [{kept:,} processed] {total_sec/3600:.2f} h audio  "
                    f"trimmed {trimmed_sec/60:.1f} min silence  "
                    f"({(time.time() - t0)/60:.1f} min)",
                    flush=True,
                )

            if args.max_samples and kept >= args.max_samples:
                break

    print(f"\n[+] Preprocessing finished:")
    print(f"  Kept: {kept:,} files ({total_sec/3600:.2f} hours)")
    print(f"  Dropped: {dropped:,} files")
    print(f"  Trimmed silence: {trimmed_sec/60:.1f} minutes")
    print(f"  Manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
