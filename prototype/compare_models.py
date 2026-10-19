"""End-to-end comparison of model combinations on the held-out photos.

usage: python compare_models.py reader_a.pt:head_a.pt [reader_b.pt:head_b.pt ...]
"""
import json
import sys
from collections import Counter
from check import Reader, check_photo
from e2e import score
from realset import TEST_PHOTOS

truth = {k: v for k, v in json.load(open("testset.json")).items() if k in TEST_PHOTOS}
for pair in sys.argv[1:]:
    weights, head = pair.split(":")
    reader, c = Reader(weights=weights, hand_head=head), Counter()
    for f, tables in truth.items():
        score(f, tables, check_photo(f"../{f}", reader)[0], c, verbose=False)
    n = sum(v for k, v in c.items() if isinstance(k, tuple))
    review = sum(v for k, v in c.items() if isinstance(k, tuple) and k[1] == "review")
    bad = c[("child right", "wrong")] + c[("child wrong", "right")]
    print(f"{weights:14} + {head:16}: held-out checked automatically {n - review}/{n} ({(n - review) / n:.0%}), "
          f"mis-marked {bad}", flush=True)
