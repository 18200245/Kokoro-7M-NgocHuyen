#!/usr/bin/env python3
"""Export flat student checkpoint into nested format compatible with Kokoro KModel.
Flat checkpoint format:
    bert.embeddings...
    decoder.conv_in...
Nested checkpoint format:
    {"bert": {...}, "decoder": {...}, "predictor": {...}, "text_encoder": {...}, "bert_encoder": {...}}
"""
import argparse
import os
import sys
import torch


def nest(flat_state_dict):
    out = {}
    for k, v in flat_state_dict.items():
        if "." not in k:
            out[k] = v
            continue
        blk, rest = k.split(".", 1)
        out.setdefault(blk, {})[rest] = v
    return out


def main():
    parser = argparse.ArgumentParser(description="Export student checkpoint to nested KModel format")
    parser.add_argument("--src", default="runs/kokoro_vi_7m/student.pth", help="Source flat checkpoint (.pth or .pt)")
    parser.add_argument("--dst", default="kokoro_vi_7m.pth", help="Destination nested checkpoint (.pth)")
    args = parser.parse_args()

    if not os.path.exists(args.src):
        # Check if last.pt exists
        alt_src = os.path.join(os.path.dirname(args.src), "last.pt")
        if os.path.exists(alt_src):
            args.src = alt_src
        else:
            print(f"Error: Source checkpoint {args.src} does not exist", file=sys.stderr)
            sys.exit(1)

    print(f"[*] Loading checkpoint from {args.src}...")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)
    flat = ckpt.get("model", ckpt)

    # Check if already nested
    if any(isinstance(v, dict) for v in flat.values()):
        print(f"[*] Checkpoint is already nested.")
        nested = flat
    else:
        nested = nest(flat)

    os.makedirs(os.path.dirname(os.path.abspath(args.dst)), exist_ok=True)
    torch.save(nested, args.dst)
    print(f"[+] Successfully exported nested model to {args.dst} (blocks: {list(nested.keys())})")


if __name__ == "__main__":
    main()
