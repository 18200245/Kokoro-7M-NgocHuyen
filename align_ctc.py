#!/usr/bin/env python3
"""Viterbi forced alignment with the trained CTC aligner -> per-phoneme durations on 40 Hz grid."""
from __future__ import annotations
import _env  # noqa: F401
import argparse
import json
import os
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.functional as AF

from train_aligner import Aligner, melspec, load_rows, BLANK, HOP, SR

PAUSE = None          # set in main(): ids whose phoneme is silence-like


def segment(ids, dur, seg_max, seg_min, pause):
    """Cut one aligned recording into training segments at pauses found by aligner.
    If seg_max == 0, the recording is returned whole.
    """
    total = int(sum(dur))
    if not seg_max or total <= seg_max:
        return [{"start": 0, "ids": list(ids), "dur": list(dur), "frames": total}]
    segs, i, pos = [], 0, 0
    n = len(dur)
    while i < n:
        acc, j, best, best_d = 0, i, -1, 0
        while j < n and acc + dur[j] <= seg_max:
            acc += dur[j]
            if ids[j] in pause and dur[j] >= best_d:
                best, best_d = j, dur[j]
            j += 1
        if j >= n:                                   # tail
            cut = n
        elif best > i:                               # cut at widest pause
            cut = best + 1
        else:                                        # hard cut
            cut = max(j, i + 1)
        sd = list(dur[i:cut])
        si = list(ids[i:cut])
        f = int(sum(sd))
        if f >= seg_min:
            segs.append({"start": pos, "ids": si, "dur": sd, "frames": f})
        pos += f
        i = cut
    return segs


def main():
    global PAUSE
    ap = argparse.ArgumentParser(description="Forced Alignment using CTC model")
    ap.add_argument("--ckpt", default="runs/aligner_vi/last.pt", help="Path to trained aligner checkpoint")
    ap.add_argument("--max-frames", type=int, default=1000)
    ap.add_argument("--out", default="data/index_raw.jsonl", help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Align only first N clips (useful for probe)")
    ap.add_argument("--manifest", default="data/vi_manifest.jsonl")
    ap.add_argument("--config", default="config_student_wide.json")
    ap.add_argument("--seg-max", type=int, default=800, help="Max frames per segment (0 for whole clip)")
    ap.add_argument("--seg-min", type=int, default=60, help="Min frames per segment (default: 60 = 1.5s)")
    ap.add_argument("--shift", type=int, default=0, help="Boundary frame shift correction")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    cfg_path = a.config if os.path.exists(a.config) else "config_student_wide.json"
    voc = json.load(open(cfg_path, encoding="utf-8"))["vocab"]
    PAUSE = {voc[c] for c in " ,.;:!?،؛؟" if c in voc}

    rows = load_rows(a.max_frames, a.manifest)
    if a.limit > 0:
        rows = rows[: a.limit]

    mel = melspec(dev)
    model = Aligner().to(dev).eval()
    print(f"[*] Loading aligner checkpoint: {a.ckpt}")
    ckpt_data = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt_data.get("model", ckpt_data)
    model.load_state_dict(state_dict)
    print(f"[*] Aligning {len(rows):,} clips (shift={a.shift}, seg_max={a.seg_max})...", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = open(a.out, "w", encoding="utf-8")
    ok = bad = 0
    scores = []

    for k, r in enumerate(rows):
        try:
            y, _ = sf.read(r["wav"], dtype="float32")
            if y.ndim > 1:
                y = y.mean(axis=1)
            n = min(len(y) // HOP, a.max_frames)
            y = y[HOP // 2 : HOP // 2 + n * HOP]
            with torch.no_grad():
                m = torch.log(mel(torch.from_numpy(np.ascontiguousarray(y))[None].to(dev)) + 1e-6)
                m = (m - m.mean()) / (m.std() + 1e-5)
                m = m[..., :n]
                lp = F.log_softmax(model(m), dim=-1)
                tgt = torch.tensor([r["ids"]], dtype=torch.long, device=dev)
                lab, sc = AF.forced_align(lp, tgt, blank=BLANK)
            spans = AF.merge_tokens(lab[0], sc[0], blank=BLANK)
            if len(spans) != len(r["ids"]):
                bad += 1
                continue
            starts = [min(max(s.start + a.shift, 0), n - 1) for s in spans]
            ends = starts[1:] + [n]
            dur = [e - s for s, e in zip(starts, ends)]
            lead, tail = starts[0], 0
            ids = [0] + list(r["ids"]) + [0]
            dur = [lead] + dur + [tail]
            keep = [(i, d) for i, d in zip(ids, dur) if d > 0]
            ids = [i for i, _ in keep]
            dur = [d for _, d in keep]
            if sum(dur) != n or len(ids) < 4:
                bad += 1
                continue
            scores.append(float(np.mean([s.score for s in spans])))
            for seg in segment(ids, dur, a.seg_max, a.seg_min, PAUSE):
                out.write(
                    json.dumps(
                        {
                            "wav": r["wav"],
                            "start": seg["start"],
                            "ids": seg["ids"],
                            "dur": seg["dur"],
                            "text": r["text"],
                            "frames": seg["frames"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                ok += 1
        except Exception:
            bad += 1

        if (k + 1) % 1000 == 0:
            out.flush()
            print(
                f"  {k+1:,}/{len(rows):,} | aligned {ok:,} | dropped {bad:,} | "
                f"mean path score: {np.mean(scores) if scores else 0:.3f}",
                flush=True,
            )

    out.close()
    mean_sc = np.mean(scores) if scores else 0.0
    print(f"[+] Aligned {ok:,} segments | Dropped {bad:,} | Mean score: {mean_sc:.3f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
