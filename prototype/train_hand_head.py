"""Train the handwriting head on PaddleOCR's frozen features (training photos, synthetic and augmented cells).

Handwritten cells are sampled 5x as often as printed ones and real cells 3x as often as generated ones;
on the held-out photos this beat equal weighting (88.7% -> 93.0% handwriting).
usage: python train_hand_head.py <feats.npz> [out=hand_head.pt] [aug_feats.npz]
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from hand_reader import HAND_ALPHABET, HandHead

STEPS, BATCH, HAND_WEIGHT, REAL_WEIGHT, LAYERS = 10000, 128, 5.0, 3.0, 2

torch.manual_seed(0)
rng = np.random.default_rng(0)
d = np.load(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else "hand_head.pt"
X = [d["train_x"], d["synth_x"]]
Y = list(d["train_y"]) + list(d["synth_y"])
K = [d["train_kind"], np.full(len(d["synth_y"]), "synth")]
if len(sys.argv) > 3:  # cells from augmented (bad-photo) versions of the training photos
    a = np.load(sys.argv[3])
    X.append(a["x"]); Y += list(a["y"]); K.append(a["kind"])
X, K = np.concatenate(X).astype(np.float32), np.concatenate(K)
w = np.where(K == "hand", HAND_WEIGHT, 1.0)
w[: len(d["train_y"])] *= REAL_WEIGHT
p = w / w.sum()

dev = "mps" if torch.backends.mps.is_available() else "cpu"
head = HandHead(layers=LAYERS).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
for step in range(STEPS):
    idx = rng.choice(len(X), BATCH, p=p)
    x = torch.from_numpy(X[idx]).to(dev)
    x = x + 0.05 * torch.randn_like(x)  # light feature noise
    tg = [torch.tensor([HAND_ALPHABET.index(c) + 1 for c in Y[i]]) for i in idx]
    lp = head(x).log_softmax(-1).permute(1, 0, 2).cpu()
    loss = F.ctc_loss(lp, torch.cat(tg), torch.full((BATCH,), lp.shape[0], dtype=torch.long),
                      torch.tensor([len(t) for t in tg]), blank=0, zero_infinity=True)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
torch.save(head.cpu().state_dict(), out)
print("saved", out)
