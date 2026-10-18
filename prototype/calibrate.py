"""How sure must the reader be before we trust it? Accuracy vs. how many cells go to teacher review.

usage: python calibrate.py [weights]   (reports on the held-out test photos)
"""
import sys
import numpy as np
import torch
from model import decode, load_reader

weights = sys.argv[1] if len(sys.argv) > 1 else "reader_v2.pt"
model = load_reader(weights)
model.eval()
d = np.load("realset.npz")

for kind in ["printed", "hand"]:
    m = (d["kind"] == kind) & d["test"]
    with torch.no_grad():
        preds = decode(model(torch.from_numpy(d["x"][m])[:, None]))
    ok = np.array([p == y for (p, _), y in zip(preds, d["y"][m])])
    conf = np.array([c for _, c in preds])
    print(f"\n{kind}: {ok.mean():.1%} correct overall ({m.sum()} cells)")
    print("  threshold | sent to review | accuracy of the rest | wrong readings that slip through")
    for t in [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]:
        keep = conf >= t
        acc = ok[keep].mean() if keep.any() else float("nan")
        print(f"     {t:.2f}   |   {1 - keep.mean():6.1%}       |  {acc:6.1%}             |  {(~ok & keep).sum()}")
    wrong = [(t, y, p, c) for t, y, (p, c), o in zip(d["tag"][m], d["y"][m], preds, ok) if not o]
    for t, y, p, c in sorted(wrong, key=lambda w: -w[3])[:12]:
        print(f"    {t:22} truth {y:>4}  read {p:>5}  conf {c:.2f}")
