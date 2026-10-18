"""Train only the last layer(s) on top of PaddleOCR's frozen pretrained features; score held-out cells.

Run with the main (PyTorch) environment:  ../.venv/bin/python train_last_layer.py
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHABET = "0123456789-.×÷+"  # blank = 0
torch.manual_seed(0); np.random.seed(0)
d = np.load("paddle_feats.npz")
X = np.concatenate([np.repeat(d["train_x"], 5, axis=0), d["synth_x"]]).astype(np.float32)  # real x5 + synthetic
Y = list(np.repeat(d["train_y"], 5)) + list(d["synth_y"])
Xt, Yt, Kt = d["test_x"].astype(np.float32), d["test_y"], d["test_kind"]
dev = "mps" if torch.backends.mps.is_available() else "cpu"


class LSTMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(120, 96, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(192, len(ALPHABET) + 1)

    def forward(self, x):
        return self.fc(self.rnn(x)[0])


def decode(logits):
    out = []
    for seq in logits.argmax(-1).tolist():
        s, prev = "", 0
        for k in seq:
            if k and k != prev:
                s += ALPHABET[k - 1]
            prev = k
        out.append(s)
    return out


for name, head in [("last layer only (Linear)", nn.Linear(120, len(ALPHABET) + 1)),
                   ("small head (BiLSTM + Linear)", LSTMHead())]:
    head = head.to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=2e-3, weight_decay=1e-4)
    steps, bs = 3000, 128
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        idx = np.random.randint(0, len(X), bs)
        x = torch.from_numpy(X[idx]).to(dev)
        x = x + 0.05 * torch.randn_like(x)  # light feature noise (no image augmentation is possible here)
        tg = [torch.tensor([ALPHABET.index(c) + 1 for c in Y[i]]) for i in idx]
        lp = head(x).log_softmax(-1).permute(1, 0, 2).cpu()  # (T, B, C)
        loss = F.ctc_loss(lp, torch.cat(tg), torch.full((bs,), lp.shape[0], dtype=torch.long),
                          torch.tensor([len(t) for t in tg]), blank=0, zero_infinity=True)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    head.eval()
    with torch.no_grad():
        preds = decode(head(torch.from_numpy(Xt).to(dev)).cpu())
    for kind in ["printed", "hand"]:
        m = Kt == kind
        acc = np.mean([p == y for p, y, k in zip(preds, Yt, Kt) if k == kind])
        print(f"PaddleOCR frozen + {name:30} {kind:7}: {acc:.1%} of {m.sum()}")
