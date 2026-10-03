#!/usr/bin/env python3
"""Re-split index segments that exceed PLBERT's 512 position embeddings at pause tokens."""
import _env  # noqa: F401
import argparse
import json
import os
import sys

from align_ctc import segment

MAX_TOK = 480  # Headroom under 512 for boundary tokens


def main():
    parser = argparse.ArgumentParser(description="Resegment alignments to stay within PLBERT 512 token cap")
    parser.add_argument("src", help="Source index JSONL (e.g. data/index_raw.jsonl)")
    parser.add_argument("dst", help="Destination index JSONL (e.g. data/index_ctc.jsonl)")
    parser.add_argument("--config", default="config_student_wide.json", help="Model config with vocab")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOK, help="Max phoneme tokens per segment")
    args = parser.parse_args()

    cfg_path = args.config if os.path.exists(args.config) else "config_student_wide.json"
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    vocab = cfg["vocab"]
    pause = {vocab[c] for c in " ,.;:!?،؛؟" if c in vocab}

    out = open(args.dst, "w", encoding="utf-8")
    kept = split = dropped = 0

    with open(args.src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if len(r["ids"]) <= args.max_tokens:
                out.write(line + "\n")
                kept += 1
                continue

            target = max(60, int(r["frames"] * args.max_tokens / len(r["ids"]) * 0.9))
            base = int(r.get("start", 0))
            for s in segment(r["ids"], r["dur"], target, 60, pause):
                if len(s["ids"]) > args.max_tokens or s["frames"] < 42:
                    dropped += 1
                    continue
                out.write(
                    json.dumps(
                        {
                            "wav": r["wav"],
                            "start": base + s["start"],
                            "ids": s["ids"],
                            "dur": s["dur"],
                            "text": r["text"],
                            "frames": s["frames"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                split += 1

    out.close()
    print(f"[+] Resegment done: kept {kept:,} | re-split into {split:,} | dropped {dropped:,} -> {args.dst}", flush=True)


if __name__ == "__main__":
    main()
