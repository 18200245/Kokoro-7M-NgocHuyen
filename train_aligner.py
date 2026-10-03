#!/usr/bin/env python3
"""A CTC phoneme aligner trained directly on real audio for duration extraction.
Framing is locked to the Kokoro student's 40 Hz grid (hop 600 at 24 kHz) so a frame index
means the same thing in the aligner and student trainer.
"""
from __future__ import annotations
import _env  # noqa: F401
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SR, HOP, NMELS = 24000, 600, 80
NVOCAB = 178          # Kokoro token ids 0..177
BLANK = NVOCAB        # CTC blank gets its own class


def melspec(dev):
    import torchaudio
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=1200, hop_length=HOP, n_mels=NMELS, center=True, power=2.0
    ).to(dev)


class Aligner(nn.Module):
    def __init__(self, d=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(NMELS, d, 5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(d),
            nn.Conv1d(d, d, 5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(d),
        )
        self.rnn = nn.LSTM(d, d, num_layers=2, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d, NVOCAB + 1)

    def forward(self, mel):                    # mel [B, NMELS, T]
        x = self.conv(mel).transpose(1, 2)
        x, _ = self.rnn(x)
        return self.out(x)                     # [B, T, NVOCAB+1]


class ClipSet(Dataset):
    """Real audio paired with phoneme token ids."""

    def __init__(self, rows, max_frames):
        self.rows, self.max_frames = rows, max_frames

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        y, _ = sf.read(r["wav"], dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        n = min(len(y) // HOP, self.max_frames)
        # Centered STFT: drop half a hop to align with 40Hz frame grid
        y = y[HOP // 2: HOP // 2 + n * HOP]
        return {"y": torch.from_numpy(np.ascontiguousarray(y)), "ids": r["ids"], "n": n}


def collate(b):
    T = max(x["n"] for x in b)
    L = max(len(x["ids"]) for x in b)
    y = torch.zeros(len(b), T * HOP)
    tgt = torch.zeros(len(b), L, dtype=torch.long)
    ylen = torch.tensor([x["n"] for x in b], dtype=torch.long)
    tlen = torch.tensor([len(x["ids"]) for x in b], dtype=torch.long)
    for i, x in enumerate(b):
        y[i, : len(x["y"])] = x["y"]
        tgt[i, : len(x["ids"])] = torch.tensor(x["ids"], dtype=torch.long)
    return y, tgt, ylen, tlen


def load_rows(max_frames, manifest=None):
    """Rows of {wav, ids, text, n}."""
    rows = []
    if manifest and os.path.exists(manifest):
        with open(manifest, encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                if not l:
                    continue
                d = json.loads(l)
                p = [i for i in d["ids"] if i != 0]
                n = d.get("frames") or int(d["duration"] * SR) // HOP
                if not p or n > max_frames or n < max(len(p) + 2, 42):
                    continue
                rows.append({"wav": d["wav"], "ids": p, "text": d.get("text", ""), "n": n})
        return rows

    # Fallback to default path
    default_m = "data/vi_manifest.jsonl"
    if os.path.exists(default_m):
        return load_rows(max_frames, default_m)
    raise FileNotFoundError(f"Manifest file not found: {manifest}")


def main():
    ap = argparse.ArgumentParser(description="Train 40Hz CTC Aligner on Vietnamese manifest")
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-frames", type=int, default=1000)      # 25 s
    ap.add_argument("--manifest", default="data/vi_manifest.jsonl",
                    help="jsonl with wav/ids/duration")
    ap.add_argument("--out", default="runs/aligner_vi")
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = load_rows(a.max_frames, a.manifest)
    random.Random(1234).shuffle(rows)
    tr, ev = rows[: -a.holdout], rows[-a.holdout:]
    print(f"[*] aligner train {len(tr):,} | held out {len(ev):,}", flush=True)

    mel = melspec(dev)
    model = Aligner().to(dev)
    print(f"[*] aligner {sum(p.numel() for p in model.parameters())/1e6:.2f}M params on {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05
    )

    step = 0
    ck = out / "last.pt"
    if a.resume and ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        step = c["step"]
        print(f"[*] Resumed at step {step}", flush=True)

    dl = DataLoader(
        ClipSet(tr, a.max_frames),
        batch_size=a.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=a.num_workers,
        drop_last=True,
        pin_memory=(dev == "cuda"),
    )

    t0 = time.time()
    acc = 0.0
    nacc = 0
    model.train()

    while step < a.steps:
        for y, tgt, ylen, tlen in dl:
            if step >= a.steps:
                break
            y, tgt = y.to(dev, non_blocking=True), tgt.to(dev)
            with torch.no_grad():
                m = torch.log(mel(y) + 1e-6)
                m = (m - m.mean(dim=(1, 2), keepdim=True)) / (m.std(dim=(1, 2), keepdim=True) + 1e-5)
                m = m[..., : int(ylen.max())]

            lp = F.log_softmax(model(m), dim=-1).transpose(0, 1)      # [T, B, C]
            loss = F.ctc_loss(lp, tgt, ylen.to(dev), tlen.to(dev), blank=BLANK, zero_infinity=True)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1

            acc += float(loss.item())
            nacc += 1

            if step % 200 == 0:
                elapsed = time.time() - t0
                speed = (step * a.batch) / max(elapsed, 1.0)
                print(
                    f"  step {step}/{a.steps} | CTC Loss: {acc/nacc:.4f} | "
                    f"lr: {sched.get_last_lr()[0]:.2e} | {speed:.1f} clip/s",
                    flush=True,
                )
                acc = 0.0
                nacc = 0

            if step % 2000 == 0 or step == a.steps:
                tmp = out / "last.pt.tmp"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "step": step,
                    },
                    tmp,
                )
                os.replace(tmp, ck)
                print(f"[+] Saved checkpoint to {ck} at step {step}", flush=True)

    print(f"[+] Aligner training complete: {step} steps. Checkpoint: {ck}", flush=True)


if __name__ == "__main__":
    main()
