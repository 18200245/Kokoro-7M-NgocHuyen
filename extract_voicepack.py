#!/usr/bin/env python3
"""Extract a 256-dim voicepack tensor (.pt) from reference audio files.
Voicepack format: tensor of shape [510, 1, 256] (float32)
- 510 = max phoneme sequence length
- 256 = style_dim * 2 (128 for decoder/timbre + 128 for predictor/prosody)
- First 128 dims come from style_encoder (acoustic/timbre)
- Last 128 dims come from predictor_encoder (prosody)
"""
import _env  # noqa: F401
import argparse
import math
import os
import random
import sys
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.utils.parametrizations import spectral_norm

BASE_KOKORO_V019_URL = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.pth"


class LearnedDownSample(nn.Module):
    def __init__(self, layer_type, dim_in):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == "none":
            self.conv = nn.Identity()
        elif self.layer_type == "timepreserve":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 1),
                    stride=(2, 1),
                    groups=dim_in,
                    padding=(1, 0),
                )
            )
        elif self.layer_type == "half":
            self.conv = spectral_norm(
                nn.Conv2d(
                    dim_in,
                    dim_in,
                    kernel_size=(3, 3),
                    stride=(2, 2),
                    groups=dim_in,
                    padding=1,
                )
            )
        else:
            raise RuntimeError(f"Unexpected downsample type: {self.layer_type}")

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    def __init__(self, layer_type):
        super().__init__()
        self.layer_type = layer_type

    def forward(self, x):
        if self.layer_type == "none":
            return x
        elif self.layer_type == "timepreserve":
            return F.avg_pool2d(x, (2, 1))
        elif self.layer_type == "half":
            if x.shape[-1] % 2 != 0:
                x = torch.cat([x, x[..., -1].unsqueeze(-1)], dim=-1)
            return F.avg_pool2d(x, 2)
        else:
            raise RuntimeError(f"Unexpected downsample type: {self.layer_type}")


class ResBlk(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        actv=nn.LeakyReLU(0.2),
        normalize=False,
        downsample="none",
    ):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = DownSample(downsample)
        self.downsample_res = LearnedDownSample(downsample, dim_in)
        self.learned_sc = dim_in != dim_out
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=True)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=True)
        if self.learned_sc:
            self.conv1x1 = spectral_norm(
                nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
            )

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = self.downsample(x)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        x = self.downsample_res(x)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)


class StyleEncoder(nn.Module):
    def __init__(self, dim_in=48, style_dim=48, max_conv_dim=384):
        super().__init__()
        blocks = []
        blocks += [spectral_norm(nn.Conv2d(1, dim_in, 3, 1, 1))]
        repeat_num = 4
        for _ in range(repeat_num):
            dim_out = min(dim_in * 2, max_conv_dim)
            blocks += [ResBlk(dim_in, dim_out, downsample="half")]
            dim_in = dim_out
        blocks += [nn.LeakyReLU(0.2)]
        blocks += [spectral_norm(nn.Conv2d(dim_out, dim_out, 5, 1, 0))]
        blocks += [nn.AdaptiveAvgPool2d(1)]
        blocks += [nn.LeakyReLU(0.2)]
        self.shared = nn.Sequential(*blocks)
        self.unshared = nn.Linear(dim_out, style_dim)

    def forward(self, x):
        h = self.shared(x)
        h = h.view(h.size(0), -1)
        s = self.unshared(h)
        return s


def ensure_model(model_path: str | None) -> str:
    if model_path and os.path.exists(model_path):
        return model_path

    target = model_path or "kokoro-v0_19.pth"
    if not os.path.exists(target):
        print(f"[*] Downloading base Kokoro model for voice extraction: {BASE_KOKORO_V019_URL}")
        urllib.request.urlretrieve(BASE_KOKORO_V019_URL, target)
        print(f"[*] Downloaded to {target}")
    return target


def extract_voicepack(
    model_path: str | None,
    audio_dir: str,
    output_path: str,
    num_samples: int = 200,
    device: str = "auto",
):
    resolved_model = ensure_model(model_path)
    dev = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    print(f"[*] Using device: {dev}")

    # Gather audio
    audio_path = Path(audio_dir)
    audio_files = sorted(list(audio_path.glob("*.flac")) + list(audio_path.glob("*.wav")))
    if not audio_files:
        print(f"[-] Error: No .wav or .flac files found in {audio_dir}", file=sys.stderr)
        sys.exit(1)

    rng = random.Random(42)
    if len(audio_files) > num_samples:
        audio_files = rng.sample(audio_files, num_samples)
    print(f"[*] Extracting voicepack from {len(audio_files)} utterances...")

    ckpt = torch.load(resolved_model, map_location="cpu", weights_only=False)
    net = ckpt.get("net", ckpt)

    def strip_prefix(sd, prefix="module."):
        return {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in sd.items()}

    style_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)
    predictor_encoder = StyleEncoder(dim_in=64, style_dim=128, max_conv_dim=512)

    se_weights = net.get("style_encoder", {})
    if not se_weights:
        # Check if flat keys
        se_weights = {k.replace("style_encoder.", ""): v for k, v in net.items() if "style_encoder." in k}
    style_encoder.load_state_dict(strip_prefix(se_weights))

    pe_weights = net.get("predictor_encoder", {})
    if not pe_weights:
        pe_weights = {k.replace("predictor_encoder.", ""): v for k, v in net.items() if "predictor_encoder." in k}

    try:
        if pe_weights:
            predictor_encoder.load_state_dict(strip_prefix(pe_weights))
        else:
            predictor_encoder.load_state_dict(strip_prefix(se_weights))
    except Exception:
        predictor_encoder.load_state_dict(strip_prefix(se_weights))

    style_encoder = style_encoder.to(dev).eval()
    predictor_encoder = predictor_encoder.to(dev).eval()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=2048,
        win_length=1200,
        hop_length=300,
        n_mels=80,
    ).to(dev)

    acoustic_vecs = []
    prosodic_vecs = []

    with torch.no_grad():
        for p in audio_files:
            try:
                y, sr = sf.read(str(p), dtype="float32")
                if y.ndim > 1:
                    y = y.mean(axis=1)
                wav = torch.from_numpy(y).unsqueeze(0).to(dev)
                if sr != 24000:
                    wav = torchaudio.transforms.Resample(sr, 24000).to(dev)(wav)

                mel = mel_transform(wav)
                mel = (torch.log(1e-5 + mel) - (-4.0)) / 4.0
                mel = mel.unsqueeze(1)  # [1, 1, n_mels, T]

                s_a = style_encoder(mel)
                s_p = predictor_encoder(mel)

                acoustic_vecs.append(s_a.cpu())
                prosodic_vecs.append(s_p.cpu())
            except Exception as e:
                continue

    if not acoustic_vecs:
        print(f"[-] Error: failed to process any audio files.", file=sys.stderr)
        sys.exit(1)

    mean_acoustic = torch.cat(acoustic_vecs, dim=0).mean(dim=0, keepdim=True)  # [1, 128]
    mean_prosodic = torch.cat(prosodic_vecs, dim=0).mean(dim=0, keepdim=True)  # [1, 128]

    # Combine to 256: first 128 acoustic, second 128 prosodic
    combined = torch.cat([mean_acoustic, mean_prosodic], dim=-1)  # [1, 256]

    # Expand to 510 length matching Kokoro voicepack standard [510, 1, 256]
    voicepack = combined.unsqueeze(0).repeat(510, 1, 1)  # [510, 1, 256]

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(voicepack, str(out_p))
    print(f"[+] Saved voicepack to {output_path} (shape: {list(voicepack.shape)})")


def main():
    parser = argparse.ArgumentParser(description="Extract voicepack (.pt) for Kokoro")
    parser.add_argument("--model", default=None, help="Base model checkpoint (.pth). Auto-downloads if omitted.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing audio clips (.flac/.wav)")
    parser.add_argument("--output", default="vi_voice.pt", help="Path to save output voicepack (.pt)")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of samples to average")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()
    extract_voicepack(
        model_path=args.model,
        audio_dir=args.audio_dir,
        output_path=args.output,
        num_samples=args.num_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
