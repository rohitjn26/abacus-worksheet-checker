"""Train the cell reader on synthetic data; report accuracy on the real labelled cells as it goes."""
import random
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from model import CRNN, decode
from norm import ALPHABET
from synth import Synth

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
BATCH = 128


class SynthStream(IterableDataset):
    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        seed = (info.id if info else 0) * 9973 + int(time.time())
        random.seed(seed); np.random.seed(seed % 2**32)
        s = Synth()
        while True:
            yield s.sample()


def collate(batch):
    x = torch.from_numpy(np.stack([b[0] for b in batch]))[:, None]
    targets = [torch.tensor([ALPHABET.index(c) + 1 for c in b[1]], dtype=torch.long) for b in batch]
    lengths = torch.tensor([len(t) for t in targets])
    return x, torch.cat(targets) if lengths.sum() else torch.zeros(0, dtype=torch.long), lengths, [b[1] for b in batch]


def evaluate(model, dev, xs, ys):
    model.eval()
    with torch.no_grad():
        preds = decode(model(torch.from_numpy(xs)[:, None].to(dev)).cpu())
    model.train()
    return sum(p == y for (p, _), y in zip(preds, ys)) / len(ys), preds


def main():
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    real = np.load("realset.npz")
    model = CRNN().to(dev)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M, device {dev}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=STEPS, pct_start=0.05)
    loader = DataLoader(SynthStream(), batch_size=BATCH, num_workers=8, collate_fn=collate, persistent_workers=True)
    t0, run = time.time(), []
    for step, (x, tg, ln, texts) in enumerate(loader, 1):
        lp = model(x.to(dev))
        loss = F.ctc_loss(lp.cpu(), tg, torch.full((x.shape[0],), lp.shape[0], dtype=torch.long), ln,
                          blank=0, zero_infinity=True)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
        run.append(loss.item())
        if step % 1000 == 0 or step == STEPS:
            p_acc, _ = evaluate(model, dev, real["p_x"], real["p_y"])
            h_acc, _ = evaluate(model, dev, real["h_x"], real["h_y"])
            print(f"step {step}  loss {np.mean(run):.3f}  real printed {p_acc:.1%}  real handwritten {h_acc:.1%}"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            run = []
            torch.save(model.state_dict(), "reader.pt")
        if step >= STEPS:
            break


if __name__ == "__main__":
    main()
