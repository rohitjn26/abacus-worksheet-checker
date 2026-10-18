"""Train the handwriting head on PaddleOCR's frozen features (training photos + synthetic cells only).

Features come from paddle_exp/extract_feats.py (same numbers as the ONNX model in hand_reader.py).
usage: python train_hand_head.py <paddle_feats.npz> [out=hand_head.pt]
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from hand_reader import HAND_ALPHABET, HandHead

torch.manual_seed(0); np.random.seed(0)
d = np.load(sys.argv[1])
out = sys.argv[2] if len(sys.argv) > 2 else "hand_head.pt"
X = np.concatenate([np.repeat(d["train_x"], 5, axis=0), d["synth_x"]]).astype(np.float32)  # real x5 + synthetic
Y = list(np.repeat(d["train_y"], 5)) + list(d["synth_y"])
dev = "mps" if torch.backends.mps.is_available() else "cpu"
head = HandHead().to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
steps, bs = 3000, 128
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
for step in range(steps):
    idx = np.random.randint(0, len(X), bs)
    x = torch.from_numpy(X[idx]).to(dev)
    x = x + 0.05 * torch.randn_like(x)  # light feature noise
    tg = [torch.tensor([HAND_ALPHABET.index(c) + 1 for c in Y[i]]) for i in idx]
    lp = head(x).log_softmax(-1).permute(1, 0, 2).cpu()
    loss = F.ctc_loss(lp, torch.cat(tg), torch.full((bs,), lp.shape[0], dtype=torch.long),
                      torch.tensor([len(t) for t in tg]), blank=0, zero_infinity=True)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
torch.save(head.cpu().state_dict(), out)
print("saved", out)
