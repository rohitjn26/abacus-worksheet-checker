"""Fine-tune the reader on real cells from the training photos, mixed with synthetic data.

Accuracy is reported on the held-out test photos only (never trained on).
"""
import random
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import CRNN
from train import SynthStream, collate, evaluate

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
BATCH, REAL_PER_BATCH = 128, 40


def augment(x):
    """Random small geometric / ink changes to a normalized (32, 128) cell."""
    h, w = x.shape
    s = random.uniform(0.85, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-4, 4), s)
    M[0, 1] += random.uniform(-0.25, 0.25)  # shear
    M[:, 2] += [random.uniform(-6, 6), random.uniform(-2, 2)]
    x = cv2.warpAffine(x, M, (w, h))
    k = random.choice([None, None, "thin", "thick"])
    if k == "thick":
        x = cv2.dilate(x, np.ones((2, 2), np.uint8))
    elif k == "thin":
        x = cv2.erode(x, np.ones((2, 2), np.uint8))
    x = x * random.uniform(0.6, 1.0)
    if random.random() < 0.3:
        x = cv2.GaussianBlur(x, (3, 3), 0)
    return np.clip(x + np.random.normal(0, 0.03, x.shape), 0, 1).astype(np.float32)


def as_list(batch):
    return batch


def report(model, dev, d, label):
    for kind in ["printed", "hand"]:
        m = (d["kind"] == kind) & d["test"]
        acc, _ = evaluate(model, dev, d["x"][m], d["y"][m])
        print(f"  {label} test {kind:7}: {acc:.1%} of {m.sum()}", flush=True)


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    d = dict(np.load("realset.npz"))  # load into memory once (an NpzFile re-reads the file on every access)
    train_idx = np.where(~d["test"])[0]
    model = CRNN().to(dev)
    model.load_state_dict(torch.load("reader.pt", map_location=dev))
    report(model, dev, d, "before")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    loader = DataLoader(SynthStream(), batch_size=BATCH - REAL_PER_BATCH, num_workers=8, collate_fn=as_list,
                        persistent_workers=True)
    t0 = time.time()
    for step, synth in enumerate(loader, 1):
        real = [(augment(d["x"][i]), str(d["y"][i])) for i in np.random.choice(train_idx, REAL_PER_BATCH)]
        x, tg, ln, _ = collate(synth + real)
        lp = model(x.to(dev))
        loss = F.ctc_loss(lp.cpu(), tg, torch.full((x.shape[0],), lp.shape[0], dtype=torch.long), ln,
                          blank=0, zero_infinity=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        if step % 1000 == 0 or step == STEPS:
            print(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
        if step >= STEPS:
            break
    torch.save(model.state_dict(), "reader_ft.pt")
    report(model, dev, d, "after")


if __name__ == "__main__":
    main()
