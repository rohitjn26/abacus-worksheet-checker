"""Can PaddleOCR (frozen) + one head read everything our own model reads? Train a head on the full alphabet
(digits, - . × ÷ +, I V X, A N S B C) and score held-out cells by kind.

Run with the main environment:  ../.venv/bin/python train_full_head.py
"""
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "../prototype")
from hand_reader import HandReader  # noqa: E402  (ONNX feature extractor)

ABC = "0123456789-.×÷+IVXANSBC"
fx = HandReader(onnx_path="../prototype/ppocrv5_mobile_frozen.onnx", head_path="../prototype/hand_head.pt")


def load(list_file, n=None):
    rows = [l.split("\t") for l in open(f"cells/{list_file}").read().splitlines() if l][:n]
    crops = [cv2.imread(f"cells/{r[0]}", cv2.IMREAD_GRAYSCALE) for r in rows]
    feats = np.concatenate([fx.features(crops[i : i + 64]) for i in range(0, len(crops), 64)])
    return feats, [r[1] for r in rows], [r[2] if len(r) > 2 else "synth" for r in rows]


parts = {k: load(f, n) for k, f, n in [("train", "train.txt", None), ("train_lab", "train_labels.txt", None),
                                      ("synth", "synth.txt", 6000), ("synth2", "synth2.txt", None),
                                      ("test", "test.txt", None), ("test_lab", "test_labels.txt", None)]}
X = np.concatenate([np.repeat(parts["train"][0], 5, 0), np.repeat(parts["train_lab"][0], 5, 0),
                    parts["synth"][0], parts["synth2"][0]]).astype(np.float32)
Y = (list(np.repeat(parts["train"][1], 5)) + list(np.repeat(parts["train_lab"][1], 5))
     + parts["synth"][1] + parts["synth2"][1])


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(120, 96, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(192, len(ABC) + 1)

    def forward(self, x):
        return self.fc(self.rnn(x)[0])


def decode(logits):
    out = []
    for seq in logits.argmax(-1).tolist():
        s, prev = "", 0
        for k in seq:
            if k and k != prev:
                s += ABC[k - 1]
            prev = k
        out.append(s)
    return out


torch.manual_seed(0); np.random.seed(0)
dev = "mps" if torch.backends.mps.is_available() else "cpu"
head = Head().to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
steps, bs = 4000, 128
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
for step in range(steps):
    idx = np.random.randint(0, len(X), bs)
    x = torch.from_numpy(X[idx]).to(dev)
    x = x + 0.05 * torch.randn_like(x)
    tg = [torch.tensor([ABC.index(c) + 1 for c in Y[i]], dtype=torch.long) for i in idx]
    ln = torch.tensor([len(t) for t in tg])
    lp = head(x).log_softmax(-1).permute(1, 0, 2).cpu()
    loss = F.ctc_loss(lp, torch.cat(tg) if ln.sum() else torch.zeros(0, dtype=torch.long),
                      torch.full((bs,), lp.shape[0], dtype=torch.long), ln, blank=0, zero_infinity=True)
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
head.eval()
torch.save(head.cpu().state_dict(), "full_head.pt")
for split in ["test", "test_lab"]:
    f, y, k = parts[split]
    with torch.no_grad():
        preds = decode(head(torch.from_numpy(f.astype(np.float32))))
    for kind in sorted(set(k)):
        sel = [(p, t) for p, t, kk in zip(preds, y, k) if kk == kind]
        print(f"PaddleOCR + full head  {kind:7}: {np.mean([p == t for p, t in sel]):.1%} of {len(sel)}",
              " misreads:", [(t, p) for p, t in sel if p != t][:8])
