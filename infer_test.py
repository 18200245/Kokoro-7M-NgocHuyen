#!/usr/bin/env python3
"""Run test inference on Vietnamese text using the distilled Kokoro 7.48M student model."""
import _env  # noqa: F401
import argparse
import os
import sys
import soundfile as sf
import torch
from kokoro import KModel
from vig2p import VietnameseG2P
from kokoro_symbols import dicts


def main():
    parser = argparse.ArgumentParser(description="Test Vietnamese TTS synthesis with Kokoro-7M")
    parser.add_argument("--text", default="Xin chào, đây là giọng đọc thử nghiệm của mô hình cô cô rô bảy mê ga byte.", help="Text to speak")
    parser.add_argument("--model", default="kokoro_vi_7m.pth", help="Path to nested model checkpoint (.pth)")
    parser.add_argument("--config", default="config_student_wide.json", help="Path to model config")
    parser.add_argument("--voicepack", default="vi_voice.pt", help="Path to voicepack (.pt)")
    parser.add_argument("--out", default="test_sample.wav", help="Path to output WAV file")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed multiplier")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"[*] Running inference on {dev}...")

    if not os.path.exists(args.model):
        print(f"[-] Error: model file {args.model} not found. Run export_model.py first.", file=sys.stderr)
        sys.exit(1)

    print(f"[*] Loading KModel from {args.model}...")
    model = KModel(
        repo_id="oddadmix/Kokoro-7M-Distill",
        config=args.config,
        model=args.model,
        disable_complex=True,
    ).eval().to(dev)

    print(f"[*] Loading voicepack from {args.voicepack}...")
    pack = torch.load(args.voicepack, map_location="cpu", weights_only=False)
    if pack.dim() == 3:
        # Shape [510, 1, 256] -> select style vector by index
        def get_style(seq_len):
            idx = min(max(seq_len - 3, 0), pack.shape[0] - 1)
            return pack[idx].to(dev)
    else:
        # Shape [1, 256] or [256]
        vec = pack.view(1, 256).to(dev)
        def get_style(seq_len):
            return vec

    print(f"[*] G2P converting text: '{args.text}'")
    g2p = VietnameseG2P()
    phonemes = g2p(args.text)
    print(f"[*] Phonemes: {phonemes}")

    tokens = [dicts[c] for c in phonemes if c in dicts]
    ref_s = get_style(len(tokens))

    with torch.no_grad():
        # Using forward_with_tokens for exact token ID control
        input_ids = torch.LongTensor([[0, *tokens, 0]]).to(dev)
        audio, dur = model.forward_with_tokens(input_ids, ref_s, speed=args.speed)
        audio_np = audio.detach().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    sf.write(args.out, audio_np, 24000)
    dur_sec = len(audio_np) / 24000
    print(f"[+] Synthesized {dur_sec:.2f}s of audio to: {args.out}")


if __name__ == "__main__":
    main()
