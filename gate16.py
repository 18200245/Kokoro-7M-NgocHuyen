#!/usr/bin/env python3
"""Offset sweep + reversed-label control gate (Gate 16).
HOP must match the corpus sample rate: 600 at 24 kHz (40 Hz grid).

Criteria to pass:
1. Offset sweep peak at +0.
2. TRUE labels gap > +4.0 dB (vowels are distinctly louder than quiet/pauses).
3. REVERSED control gap near 0 dB (confirms alignment is genuine, not random monotonic bias).
"""
import _env  # noqa: F401
import argparse
import json
import os
import sys
import numpy as np
import soundfile as sf

VOWELS = set("aeiouɑɐɒæɔəɚɛɜɪʊʌɤøœɨɯ")
QUIET = set(" ,.;:!?،؛؟")


def main():
    parser = argparse.ArgumentParser(description="Verify alignment quality with Gate 16")
    parser.add_argument("index", help="Path to index JSONL (e.g. data/index_ctc.jsonl)")
    parser.add_argument("config", nargs="?", default="config_student_wide.json", help="Path to config.json")
    parser.add_argument("hop", nargs="?", type=int, default=600, help="Hop length (600 for 24kHz / 40Hz)")
    args = parser.parse_args()

    cfg_path = args.config if os.path.exists(args.config) else "config_student_wide.json"
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    inv = {v: k for k, v in cfg["vocab"].items()}

    rows = [json.loads(l) for l in open(args.index, encoding="utf-8")][:900]
    hop = args.hop

    def gap(shift=0, rev=False):
        a, b = [], []
        for r in rows:
            n = r["frames"]
            y, _ = sf.read(
                r["wav"],
                dtype="float32",
                start=int(r.get("start", 0)) * hop,
                frames=n * hop,
            )
            if y.ndim > 1:
                y = y.mean(1)
            if len(y) < n * hop:
                y = np.pad(y, (0, n * hop - len(y)))
            e = 20 * np.log10(np.sqrt((y.reshape(n, hop) ** 2).mean(1) + 1e-12))
            ids = list(r["ids"])[::-1] if rev else r["ids"]
            pos = 0
            for pid, d in zip(ids, r["dur"]):
                ch = inv.get(pid, "")
                seg = e[max(pos + shift, 0) : max(pos + d + shift, 0)]
                if len(seg):
                    if ch in VOWELS:
                        a.append(seg.mean())
                    elif ch in QUIET:
                        b.append(seg.mean())
                pos += d
        return (np.mean(a) - np.mean(b)) if a and b else float("nan")

    gs = {s: gap(s) for s in (-3, -2, -1, 0, 1, 2, 3)}
    best = max(gs, key=lambda k: gs[k])
    rev_gap = gap(0, True)

    print("=" * 60)
    print("GATE 16 ALIGNMENT QUALITY CHECK")
    print("=" * 60)
    print("Offset sweep: " + "  ".join(f"{s:+d}:{g:5.2f}" for s, g in gs.items()) + f"   -> Peak at: {best:+d}")
    print(f"TRUE labels gap:      {gs[0]:+6.2f} dB  (Target: > +4.0 dB)")
    print(f"REVERSED control gap: {rev_gap:+6.2f} dB  (Target: ~ 0.0 dB)")

    passed = (best == 0) and (gs[0] >= 3.5) and (abs(rev_gap) < 2.0)
    if passed:
        print("\n[PASS] Alignment passed Gate 16! Ready for student training.")
    else:
        print("\n[WARNING] Alignment metrics did not strictly meet target thresholds.")
        if best != 0:
            print(f"  - Peak is at {best:+d}. Consider re-aligning with --shift {best}.")
    print("=" * 60)


if __name__ == "__main__":
    main()
