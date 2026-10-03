#!/usr/bin/env python3
"""Stage 4: Distill / Finetune 7.48M Kokoro student for Vietnamese.

Architecture:
- PLBERT: Hidden 192, 12 layers, 3 heads
- Hidden Dim: 160
- Decoder: 2 resblocks [3, 7], init channels 128
- Vocab: 178 tokens
- Discriminators: MPD + MSD (StyleTTS2) + optional WavLM SLM loss
"""
from __future__ import annotations
import _env  # noqa: F401
import argparse
import json
import math
import os
import random
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AlbertConfig

from kokoro.istftnet import Decoder
from kokoro.modules import CustomAlbert, ProsodyPredictor, TextEncoder
from styletts2_gan.discriminators import (
    MultiPeriodDiscriminator,
    MultiResSpecDiscriminator,
    WavLMDiscriminator,
)
from styletts2_gan.losses_styletts2 import (
    DiscriminatorLoss,
    GeneratorLoss,
    MultiResolutionSTFTLoss,
    WavLMLoss,
)

EN_7M_URL = "https://huggingface.co/oddadmix/Kokoro-7M-Distill/resolve/main/kokoro_en_7m.pth"


class MelL1(nn.Module):
    """L1 on log-mel for consonant / F2-F3 detail (1-4 kHz)."""

    def __init__(self, sr=24000, n_fft=1024, hop=256, n_mels=80):
        super().__init__()
        import torchaudio
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=1.0
        )

    def forward(self, y_hat, y):
        a = torch.log(self.mel(y_hat) + 1e-5)
        b = torch.log(self.mel(y) + 1e-5)
        return F.l1_loss(a, b)


SPF = 600      # 24000 Hz / 40 Hz = 600 samples per frame
FRAME_HZ = 40


def build_student(cfg_path):
    c = json.load(open(cfg_path, encoding="utf-8"))
    m = nn.Module()
    m.bert = CustomAlbert(AlbertConfig(vocab_size=c["n_token"], **c["plbert"]))
    m.bert_encoder = nn.Linear(m.bert.config.hidden_size, c["hidden_dim"])
    m.predictor = ProsodyPredictor(
        style_dim=c["style_dim"],
        d_hid=c["hidden_dim"],
        nlayers=c["n_layer"],
        max_dur=c["max_dur"],
        dropout=c["dropout"],
    )
    m.text_encoder = TextEncoder(
        channels=c["hidden_dim"],
        kernel_size=c["text_encoder_kernel_size"],
        depth=c["n_layer"],
        n_symbols=c["n_token"],
    )
    m.decoder = Decoder(
        dim_in=c["hidden_dim"],
        style_dim=c["style_dim"],
        dim_out=c["n_mels"],
        disable_complex=True,
        **c["istftnet"],
    )
    return m, c


class DistDataset(Dataset):
    """Real Vietnamese audio with aligned CTC durations."""

    def __init__(self, rows, crop):
        self.rows = rows
        self.crop = crop

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        import soundfile as sf

        need = int(sum(r["dur"])) * SPF
        off = int(r.get("start", 0)) * SPF
        a, sr = sf.read(r["wav"], dtype="float32", start=off, frames=need)
        if a.ndim > 1:
            a = a.mean(axis=1)
        if len(a) < need:
            a = np.pad(a, (0, need - len(a)))
        a = a[:need]
        return {"ids": r["ids"], "dur": r["dur"], "audio": a, "lang": "vi"}


def collate(batch):
    B = len(batch)
    L = max(len(b["ids"]) for b in batch)
    ids = torch.zeros(B, L, dtype=torch.long)
    dur = torch.zeros(B, L, dtype=torch.float)
    lens = torch.tensor([len(b["ids"]) for b in batch], dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
        dur[i, : len(b["dur"])] = torch.tensor(b["dur"], dtype=torch.float)
    return {
        "ids": ids,
        "dur": dur,
        "lens": lens,
        "lang": [b.get("lang", "vi") for b in batch],
        "audio": [torch.from_numpy(b["audio"]) for b in batch],
    }


def build_aln(dur, lens, F_max, device):
    """[B,L] integer durations -> [B,L,F] hard monotonic alignment."""
    B, L = dur.shape
    aln = torch.zeros(B, L, F_max, device=device)
    for b in range(B):
        pos = 0
        for j in range(int(lens[b])):
            d = int(dur[b, j])
            if d <= 0 or pos >= F_max:
                continue
            aln[b, j, pos : min(pos + d, F_max)] = 1.0
            pos += d
    return aln


def ensure_init_weights(init_path: str | None) -> str | None:
    if not init_path:
        return None
    if os.path.exists(init_path):
        return init_path
    if "7m" in init_path.lower():
        print(f"[*] Downloading base 7M weights: {EN_7M_URL}")
        os.makedirs(os.path.dirname(os.path.abspath(init_path)), exist_ok=True)
        urllib.request.urlretrieve(EN_7M_URL, init_path)
        print(f"[*] Downloaded to {init_path}")
        return init_path
    return init_path


def main():
    ap = argparse.ArgumentParser(description="Train 7.48M Kokoro student for Vietnamese")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=40)        # 40 frames * 600 = 1.0 s
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-disc", type=float, default=2e-4)
    ap.add_argument("--lr-decay", action="store_true")
    ap.add_argument("--dur-weight", type=float, default=5.0)
    ap.add_argument("--stft-weight", type=float, default=5.0)
    ap.add_argument("--disc-warmup", type=int, default=500)
    ap.add_argument("--slm-weight", type=float, default=1.0)
    ap.add_argument("--slm-adv-weight", type=float, default=0.2)
    ap.add_argument("--mel-weight", type=float, default=5.0)
    ap.add_argument("--sil-weight", type=float, default=100.0)
    ap.add_argument("--sil-thresh", type=float, default=1e-3)
    ap.add_argument("--init-from", default="kokoro_en_7m.pth", help="Base weights to warm start from")
    ap.add_argument("--index", default="data/index_ctc.jsonl")
    ap.add_argument("--voicepack", default="vi_voice.pt", help="Voicepack style vector (.pt)")
    ap.add_argument("--config", default="config_student_wide.json")
    ap.add_argument("--out", default="runs/kokoro_vi_7m")
    ap.add_argument("--holdout", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--sr", type=int, default=24000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(a.index, encoding="utf-8")]
    rows = [r for r in rows if sum(r["dur"]) >= a.crop + 2]
    random.Random(1234).shuffle(rows)
    train_rows, eval_rows = rows[: -a.holdout], rows[-a.holdout :]
    print(f"[*] Train samples: {len(train_rows):,} | Held out: {len(eval_rows):,}", flush=True)

    model, cfg = build_student(a.config)
    model.to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Student model initialized: {n_params/1e6:.2f}M parameters on {dev}", flush=True)

    # Load Voicepack
    vp_path = a.voicepack
    if os.path.exists(vp_path):
        pack = torch.load(vp_path, map_location="cpu", weights_only=False)
        print(f"[*] Loaded voicepack from {vp_path} (shape {list(pack.shape)})", flush=True)
    else:
        print(f"[!] Warning: Voicepack {vp_path} not found. Initializing random voicepack.", flush=True)
        pack = torch.randn(510, 1, 256)

    # Ensure shape is [N, 1, 256]
    if pack.dim() == 2:
        pack = pack.unsqueeze(1)
    if pack.shape[-1] != 256:
        raise ValueError(f"Voicepack must have style dimension 256 (got {pack.shape})")

    mpd, msd = MultiPeriodDiscriminator().to(dev), MultiResSpecDiscriminator().to(dev)
    gen_loss = GeneratorLoss(mpd, msd).to(dev)
    disc_loss = DiscriminatorLoss(mpd, msd).to(dev)
    stft_loss = MultiResolutionSTFTLoss().to(dev)
    mel_loss = MelL1(sr=a.sr).to(dev)

    wl = None
    if a.slm_weight > 0 or a.slm_adv_weight > 0:
        try:
            print("[*] Initializing WavLM discriminator and loss...")
            wd = WavLMDiscriminator(slm_hidden=768, slm_layers=13, initial_channel=64).to(dev)
            wl = WavLMLoss("microsoft/wavlm-base-plus", wd, a.sr, 16000).to(dev)
            for p_ in wl.wavlm.parameters():
                p_.requires_grad = False
            wl.wavlm.eval()
            print(f"[*] SLM Loss enabled: feature weight={a.slm_weight}, adv weight={a.slm_adv_weight}", flush=True)
        except Exception as e:
            print(f"[!] Warning: Failed to load WavLM ({e}). Disabling SLM loss.", flush=True)
            a.slm_weight = 0.0
            a.slm_adv_weight = 0.0
            wl = None

    opt_g = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.8, 0.99), weight_decay=1e-2)
    sched_g = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt_g, T_max=a.steps, eta_min=a.lr * 0.1)
        if a.lr_decay
        else None
    )
    opt_d = torch.optim.AdamW(
        list(mpd.parameters()) + list(msd.parameters()),
        lr=a.lr_disc,
        betas=(0.8, 0.99),
        weight_decay=1e-2,
    )

    dl = DataLoader(
        DistDataset(train_rows, a.crop),
        batch_size=a.batch,
        shuffle=True,
        collate_fn=collate,
        num_workers=a.num_workers,
        drop_last=True,
        pin_memory=(dev == "cuda"),
    )

    def run_batch(b, probe=False):
        ids = b["ids"].to(dev)
        lens = b["lens"].to(dev)
        durt = b["dur"].to(dev)
        B, L = ids.shape
        tmask = torch.arange(L, device=dev)[None].expand(B, -1) + 1 > lens[:, None]

        ref = torch.cat(
            [pack[min(max(int(l) - 3, 0), pack.shape[0] - 1)] for l in lens]
        ).to(dev)
        s_pred, s_dec = ref[:, 128:], ref[:, :128]

        bert_dur = model.bert(ids, attention_mask=(~tmask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
        d = model.predictor.text_encoder(d_en, s_pred, lens, tmask)
        x, _ = model.predictor.lstm(d)
        dur_pred = torch.sigmoid(model.predictor.duration_proj(x)).sum(dim=-1)
        l_dur = (F.l1_loss(dur_pred, durt, reduction="none") * (~tmask)).sum() / (~tmask).sum()

        frames = durt.sum(dim=1).long()
        Fmax = int(frames.max())
        aln = build_aln(durt, lens, Fmax, dev)
        en = d.transpose(-1, -2) @ aln
        F0, N = model.predictor.F0Ntrain(en, s_pred)
        t_en = model.text_encoder(ids, lens, tmask)
        asr = t_en @ aln
        up = round(F0.shape[-1] / asr.shape[-1])

        C = a.crop
        asr_c = torch.empty(B, asr.shape[1], C, device=dev)
        F0_c = torch.empty(B, C * up, device=dev)
        N_c = torch.empty(B, C * up, device=dev)
        y = torch.empty(B, 1, C * SPF, device=dev)
        for i in range(B):
            hi = max(int(frames[i]) - C, 0)
            f0 = random.randint(0, hi)
            asr_c[i] = asr[i, :, f0 : f0 + C]
            F0_c[i] = F0[i, f0 * up : (f0 + C) * up]
            N_c[i] = N[i, f0 * up : (f0 + C) * up]
            w = b["audio"][i]
            seg = w[f0 * SPF : (f0 + C) * SPF]
            if seg.shape[0] < C * SPF:
                seg = F.pad(seg, (0, C * SPF - seg.shape[0]))
            y[i, 0] = seg.to(dev)

        y_hat = model.decoder(asr_c, F0_c, N_c, s_dec)
        if y_hat.dim() == 2:
            y_hat = y_hat.unsqueeze(1)
        y_hat = y_hat[..., : y.shape[-1]]

        return y, y_hat, l_dur

    if a.probe:
        b = next(iter(dl))
        y, y_hat, l_dur = run_batch(b, probe=True)
        print(
            f"[+] Probe check passed: dur_L1={l_dur.item():.4f}, "
            f"stft={stft_loss(y_hat.squeeze(1), y.squeeze(1)).item():.4f}",
            flush=True,
        )
        return

    step = 0
    init_file = ensure_init_weights(a.init_from)
    if init_file and os.path.exists(init_file):
        c = torch.load(init_file, map_location="cpu", weights_only=False)
        src = c["model"] if "model" in c else c
        if src and all(isinstance(v, dict) for v in src.values()):
            src = {f"{b}.{k}": t for b, d in src.items() for k, t in d.items()}
        if any(".module." in k for k in src):
            src = {k.replace(".module.", ".", 1): v for k, v in src.items()}
        tgt = model.state_dict()
        fit = {k: v for k, v in src.items() if k in tgt and tgt[k].shape == v.shape}
        model.load_state_dict(fit, strict=False)
        n_fit = sum(tgt[k].numel() for k in fit)
        n_all = sum(v.numel() for v in tgt.values())
        print(
            f"[*] Warm-started {n_fit/1e6:.2f}M of {n_all/1e6:.2f}M "
            f"({100*n_fit/n_all:.1f}%) parameters from {init_file}",
            flush=True,
        )
        if "mpd" in c and "msd" in c:
            try:
                mpd.load_state_dict(c["mpd"])
                msd.load_state_dict(c["msd"])
            except Exception:
                pass

    ck = out / "last.pt"
    if a.resume and ck.exists():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        model.load_state_dict(c["model"])
        mpd.load_state_dict(c["mpd"])
        msd.load_state_dict(c["msd"])
        opt_g.load_state_dict(c["opt_g"])
        opt_d.load_state_dict(c["opt_d"])
        step = c["step"]
        if sched_g is not None and "sched_g" in c:
            sched_g.load_state_dict(c["sched_g"])
        print(f"[*] Resumed training from checkpoint at step {step}", flush=True)

    t0 = time.time()
    acc = {}
    model.train()

    print(f"[*] Beginning training for {a.steps} steps...")
    while step < a.steps:
        for b in dl:
            if step >= a.steps:
                break
            y, y_hat, l_dur = run_batch(b)

            # Discriminator update
            opt_d.zero_grad(set_to_none=True)
            d_l = disc_loss(y, y_hat.detach())
            d_l.backward()
            torch.nn.utils.clip_grad_norm_(list(mpd.parameters()) + list(msd.parameters()), 10.0)
            opt_d.step()

            # Generator update
            opt_g.zero_grad(set_to_none=True)
            l_stft = stft_loss(y_hat.squeeze(1), y.squeeze(1))
            l_mel = (
                mel_loss(y_hat.squeeze(1), y.squeeze(1))
                if a.mel_weight > 0
                else torch.zeros((), device=dev)
            )

            l_slm = torch.zeros((), device=dev)
            if wl is not None:
                if a.slm_weight > 0:
                    l_slm = wl(y.squeeze(1), y_hat.squeeze(1))
                if a.slm_adv_weight > 0 and step >= a.disc_warmup:
                    l_slm = l_slm + (a.slm_adv_weight / max(a.slm_weight, 1e-9)) * wl.generator(y_hat.squeeze(1))

            l_sil = torch.zeros((), device=dev)
            if a.sil_weight > 0:
                H = a.sr // 100  # 10ms frame
                n = y.shape[-1] // H
                yr = y[..., : n * H].reshape(y.shape[0], n, H)
                hr = y_hat[..., : n * H].reshape(y_hat.shape[0], n, H)
                yrms = (yr.pow(2).mean(-1) + 1e-12).sqrt()
                tsil = (yrms < a.sil_thresh).float()
                if tsil.sum() > 0:
                    hrms = (hr.pow(2).mean(-1) + 1e-12).sqrt()
                    l_sil = (F.relu(hrms - yrms) * tsil).sum() / tsil.sum()

            l_g = gen_loss(y, y_hat) if step >= a.disc_warmup else torch.zeros((), device=dev)
            loss = (
                a.stft_weight * l_stft
                + a.dur_weight * l_dur
                + l_g
                + a.sil_weight * l_sil
                + a.mel_weight * l_mel
                + a.slm_weight * l_slm
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt_g.step()
            if sched_g is not None:
                sched_g.step()
            step += 1

            for k, v in (
                ("stft", l_stft),
                ("dur", l_dur),
                ("gen", l_g),
                ("disc", d_l),
                ("sil", l_sil),
                ("mel", l_mel),
                ("slm", l_slm),
            ):
                acc[k] = acc.get(k, 0.0) + float(v.item() if hasattr(v, "item") else v)

            if step % 100 == 0:
                speed = (step * a.batch) / max(time.time() - t0, 1.0)
                stats = "  ".join(f"{k}: {v/100:.3f}" for k, v in acc.items())
                print(f"  Step {step}/{a.steps} | {stats} | {speed:.1f} clip/s", flush=True)
                acc = {}

            if a.ckpt_every and (step % a.ckpt_every == 0 or step == a.steps):
                tmp = out / "last.pt.tmp"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "mpd": mpd.state_dict(),
                        "msd": msd.state_dict(),
                        "opt_g": opt_g.state_dict(),
                        "opt_d": opt_d.state_dict(),
                        "step": step,
                        "config": cfg,
                        **({"sched_g": sched_g.state_dict()} if sched_g else {}),
                    },
                    tmp,
                )
                os.replace(tmp, ck)
                torch.save(model.state_dict(), out / "student.pth")
                print(f"[+] Checkpoint saved at step {step}: {ck}", flush=True)

    print(f"[+] Training finished successfully at step {step}! Checkpoint saved to {out}", flush=True)


if __name__ == "__main__":
    main()
