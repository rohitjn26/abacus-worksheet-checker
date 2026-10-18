"""Decide right / wrong / review for one answer, and measure the rule on the held-out test photos.

The rule:
  * reading == correct answer (and not a wild guess)          -> right
  * reading != correct answer, confidently, and the model finds
    the correct answer implausible for this handwriting          -> wrong
  * anything else                                               -> review (teacher looks)
The correct answer never creates a 'right' from an unclear reading; it only stops a doubtful 'wrong'.
"""
import json
import numpy as np
import torch
from model import string_logprob

RIGHT_MIN_CONF = 0.5      # reading equals the correct answer: accept unless the reader is basically guessing
WRONG_MIN_CONF = 0.95     # reading differs: the reader must be sure of what it read ...
ALT_MAX_RATIO = 0.001     # ... and the correct answer must be < 0.1% as likely as the reading


def decide(read, conf, correct, alt_ratio):
    """read: model's reading; correct: computed answer as text; alt_ratio: P(correct)/P(read)."""
    if not read:
        return "blank"
    if read == correct:
        return "right" if conf >= RIGHT_MIN_CONF else "review"
    if conf >= WRONG_MIN_CONF and alt_ratio < ALT_MAX_RATIO:
        return "wrong"
    return "review"


if __name__ == "__main__":
    import sys
    from decimal import Decimal
    from model import CRNN, decode
    from check import evaluate, fmt

    model = CRNN()
    model.load_state_dict(torch.load(sys.argv[1] if len(sys.argv) > 1 else "reader_ft.pt", map_location="cpu"))
    model.eval()
    d = np.load("realset.npz")
    truth = json.load(open("testset.json"))
    correct_by_tag = {}
    for fname, tables in truth.items():
        if fname.startswith("_"):
            continue
        for ti, gt in enumerate(tables):
            for qi, nums in enumerate(gt["numbers"]):
                if "?" not in nums:
                    correct_by_tag[f"{fname[-14:-5]} T{ti + 1} Q{qi + 1}"] = fmt(evaluate([str(n) for n in nums]))

    m = (d["kind"] == "hand") & d["test"] & np.array([t in correct_by_tag for t in d["tag"]])
    x = torch.from_numpy(d["x"][m])[:, None]
    with torch.no_grad():
        lp = model(x)
    preds = decode(lp)
    correct = [correct_by_tag[t] for t in d["tag"][m]]
    reads = [p if p else "0" for p, _ in preds]
    ratio = (string_logprob(lp, correct) - string_logprob(lp, reads)).exp().numpy()

    counts = {}
    for (read, conf), c, r, written, tag in zip(preds, correct, ratio, d["y"][m], d["tag"][m]):
        verdict = decide(read, conf, c, r)
        child = "child right" if written == c else "child wrong"
        counts[(child, verdict)] = counts.get((child, verdict), 0) + 1
        if (child, verdict) in [("child right", "wrong"), ("child wrong", "right")]:
            print(f"  !! {tag}: child wrote {written}, correct {c}, read {read} ({conf:.2f}) -> {verdict}")
    n = int(m.sum())
    print(f"\n{n} answers on held-out photos")
    for child in ["child right", "child wrong"]:
        row = {v: counts.get((child, v), 0) for v in ["right", "wrong", "review", "blank"]}
        print(f"  {child:12}: marked right {row['right']:3}  marked wrong {row['wrong']:3}  review {row['review']:3}")
    review = sum(v for (c, k), v in counts.items() if k == "review")
    bad = counts.get(("child right", "wrong"), 0) + counts.get(("child wrong", "right"), 0)
    print(f"  sent to teacher: {review}/{n} ({review / n:.0%}),  mis-marked: {bad}/{n}")
