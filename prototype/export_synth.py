"""Export synthetic training cells as plain images (before our normalisation), PaddleOCR list format.

usage: python export_synth.py <out_dir> <count>
"""
import random
import sys
from pathlib import Path
import cv2
import numpy as np
from synth import Synth

out, n = Path(sys.argv[1]), int(sys.argv[2])
(out / "synth").mkdir(parents=True, exist_ok=True)
s, rows = Synth(), []
allowed = set("0123456789-.×÷+")
while len(rows) < n:
    printed = random.random() < 0.45
    text = s.number(printed)
    if not set(text) <= allowed:
        continue
    ink, dark = s.printed(text) if printed else s.handwritten(text)
    name = f"synth/{len(rows):06d}.png"
    cv2.imwrite(str(out / name), s.photo(ink, dark))  # photo-like grey crop
    rows.append(f"{name}\t{text}")
(out / "synth.txt").write_text("\n".join(rows) + "\n")
print(len(rows))
