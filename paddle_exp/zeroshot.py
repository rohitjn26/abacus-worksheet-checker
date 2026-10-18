"""Zero-shot: PaddleOCR text recognizers on our held-out cells, no training on our data."""
import sys
import cv2
import numpy as np
from paddleocr import TextRecognition

DASH = str.maketrans({"−": "-", "–": "-", "—": "-", "_": "-", "一": "-"})
LENIENT = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "i": "1"})


def clean(t, lenient=False):
    t = t.replace(" ", "").translate(DASH)
    return t.translate(LENIENT) if lenient else t


root = "cells"
rows = [l.split("\t") for l in open(f"{root}/test.txt").read().splitlines()]
for name in sys.argv[1:]:
    model = TextRecognition(model_name=name)
    imgs = [cv2.cvtColor(cv2.imread(f"{root}/{p}", cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR) for p, _, _ in rows]
    preds = [r["rec_text"] for r in model.predict(input=imgs, batch_size=32)]
    for kind in ["printed", "hand"]:
        sel = [(clean(p), clean(p, True), y) for p, (_, y, k) in zip(preds, rows) if k == kind]
        exact = np.mean([a == y for a, _, y in sel]); len_ = np.mean([b == y for _, b, y in sel])
        print(f"{name:22} {kind:7}: {exact:.1%} exact, {len_:.1%} with O->0 / l->1 fixes  (n={len(sel)})")
        if kind == "hand":
            print("   examples (truth -> read):", [(y, a) for a, _, y in sel if a != y][:12])
