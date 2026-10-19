"""Calibrate the handwriting head's confidence on the calibration photos (never trained on, never the test).

Finds the temperature T (logits / T) that best fits how often the head is actually right, by minimising the
CTC negative log-likelihood of the true answers. Saves <head>.json, which hand_reader.py picks up.
usage: python calibrate_head.py <feats.npz> <head.pt>
"""
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from hand_reader import HAND_ALPHABET, HandHead
from model import decode

d = np.load(sys.argv[1])
state = torch.load(sys.argv[2], map_location="cpu")
head = HandHead(layers=2 if "rnn.weight_ih_l1" in state else 1)
head.load_state_dict(state)
head.eval()
m = d["calib_kind"] == "hand"
x, y = torch.from_numpy(d["calib_x"][m].astype(np.float32)), list(d["calib_y"][m])
with torch.no_grad():
    logits = head(x)
tg = [torch.tensor([HAND_ALPHABET.index(c) + 1 for c in s]) for s in y]


def nll(T):
    lp = (logits / T).log_softmax(-1).permute(1, 0, 2)
    return F.ctc_loss(lp, torch.cat(tg), torch.full((len(y),), lp.shape[0], dtype=torch.long),
                      torch.tensor([len(t) for t in tg]), blank=0, reduction="mean", zero_infinity=True).item()


def reliability(T):
    preds = decode((logits / T).log_softmax(-1).permute(1, 0, 2), HAND_ALPHABET)
    conf = np.array([c for _, c in preds]); ok = np.array([p == t for (p, _), t in zip(preds, y)])
    hi = conf >= 0.95
    return ok.mean(), hi.mean(), ok[hi].mean() if hi.any() else float("nan"), (~ok & hi).sum()


Ts = np.round(np.arange(0.5, 6.01, 0.1), 2)
best = min(Ts, key=nll)
for T in [1.0, best]:
    acc, cover, acc_hi, wrong_hi = reliability(T)
    print(f"T={T:.1f}: NLL {nll(T):.3f} | {len(y)} calibration answers, {acc:.1%} read right | "
          f"{cover:.0%} read with >=95% confidence, of those {acc_hi:.1%} right ({wrong_hi} wrong but 'sure')")
json.dump({"temperature": float(best)}, open(os.path.splitext(sys.argv[2])[0] + ".json", "w"))
print("saved temperature", best)
