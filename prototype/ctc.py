"""CTC decoding and string probabilities in plain numpy (no PyTorch), so the checker runs on a phone.

log_probs everywhere: numpy array (T, B, C), log-softmax over C, blank = class 0, char i = class i + 1.
"""
import numpy as np


def log_softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=axis, keepdims=True))


def _logsumexp(a, b):
    m = np.maximum(a, b)
    with np.errstate(invalid="ignore"):
        out = m + np.log(np.exp(a - m) + np.exp(b - m))
    return np.where(np.isneginf(m), -np.inf, out)


def string_logprob(log_probs, texts, alphabet):
    """log P(text | image), summed over all CTC alignments, for each column b of log_probs and texts[b]."""
    T = log_probs.shape[0]
    out = np.empty(len(texts))
    for b, text in enumerate(texts):
        lp = log_probs[:, b, :].astype(np.float64)
        labels = [alphabet.index(c) + 1 for c in text]
        ext = [0]
        for k in labels:
            ext += [k, 0]  # blank, c1, blank, c2, ..., blank
        S = len(ext)
        ext = np.array(ext)
        # a transition may skip a blank only between two different characters
        skip = np.zeros(S, bool)
        skip[2:] = (ext[2:] != 0) & (ext[2:] != ext[:-2])
        alpha = np.full(S, -np.inf)
        alpha[0] = lp[0, 0]
        if S > 1:
            alpha[1] = lp[0, ext[1]]
        for t in range(1, T):
            prev = alpha
            a = prev.copy()
            a[1:] = _logsumexp(a[1:], prev[:-1])
            a[2:] = np.where(skip[2:], _logsumexp(a[2:], prev[:-2]), a[2:])
            alpha = a + lp[t, ext]
        out[b] = _logsumexp(alpha[-1], alpha[-2]) if S > 1 else alpha[-1]
    return out


def decode(log_probs, alphabet):
    """Greedy CTC decode. Returns [(text, confidence)], confidence = P(text | image) over all alignments."""
    idx = log_probs.argmax(-1)  # (T, B)
    texts = []
    for b in range(idx.shape[1]):
        text, prev = "", 0
        for k in idx[:, b].tolist():
            if k != 0 and k != prev:
                text += alphabet[k - 1]
            prev = k
        texts.append(text)
    return list(zip(texts, np.exp(string_logprob(log_probs, texts, alphabet)).tolist()))
