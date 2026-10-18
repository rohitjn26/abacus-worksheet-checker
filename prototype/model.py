"""Small CRNN + CTC reader for one cell (printed or handwritten number). ~1M parameters."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from norm import ALPHABET


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CRNN(nn.Module):
    def __init__(self, n_classes=len(ALPHABET) + 1):
        super().__init__()
        self.cnn = nn.Sequential(
            block(1, 32), nn.MaxPool2d(2),            # 16 x 64
            block(32, 64), nn.MaxPool2d(2),           # 8 x 32
            block(64, 128), block(128, 128), nn.MaxPool2d((2, 1)),  # 4 x 32
            block(128, 192), nn.MaxPool2d((2, 1)),    # 2 x 32
            nn.Conv2d(192, 256, (2, 1)), nn.ReLU(inplace=True),  # 1 x 32
        )
        self.rnn = nn.LSTM(256, 128, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):  # x: (B, 1, 32, 128) -> log-probs (T=32, B, C)
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        f, _ = self.rnn(f)
        return self.fc(f).log_softmax(-1).permute(1, 0, 2)


def load_reader(path, device="cpu"):
    """Load weights; if they predate new alphabet characters, grow the output layer (old classes keep
    their weights, new ones start untrained)."""
    model = CRNN()
    state = torch.load(path, map_location=device)
    w, b = state["fc.weight"], state["fc.bias"]
    if w.shape[0] < model.fc.out_features:
        new_w, new_b = model.fc.weight.data.clone(), model.fc.bias.data.clone()
        new_w[: w.shape[0]], new_b[: b.shape[0]] = w, b
        new_b[b.shape[0]:] = b.min()  # new characters start unlikely
        state["fc.weight"], state["fc.bias"] = new_w, new_b
    model.load_state_dict(state)
    return model.to(device)


def string_logprob(log_probs, texts, alphabet=ALPHABET):
    """log P(text | image) under CTC (summed over all alignments), per column of log_probs (T, B, C)."""
    tg = [torch.tensor([alphabet.index(c) + 1 for c in t], dtype=torch.long) for t in texts]
    ln = torch.tensor([len(t) for t in tg])
    T, B, _ = log_probs.shape
    flat = torch.cat(tg) if ln.sum() else torch.zeros(0, dtype=torch.long)
    return -F.ctc_loss(log_probs, flat, torch.full((B,), T, dtype=torch.long), ln, blank=0,
                       reduction="none", zero_infinity=True)


def decode(log_probs, alphabet=ALPHABET):
    """Greedy CTC decode. Returns [(text, confidence)], confidence = P(text | image) over all alignments
    (a character's edges wavering between it and 'blank' does not count as doubt)."""
    idx = log_probs.argmax(-1)  # (T, B)
    texts = []
    for b in range(idx.shape[1]):
        text, prev = "", 0
        for k in idx[:, b].tolist():
            if k != 0 and k != prev:
                text += alphabet[k - 1]
            prev = k
        texts.append(text)
    conf = string_logprob(log_probs, texts, alphabet).exp().tolist()
    return list(zip(texts, conf))
