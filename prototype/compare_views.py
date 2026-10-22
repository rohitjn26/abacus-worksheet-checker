"""Score answer-view combinations end to end on every split.  usage: python compare_views.py "tight,rows,above" "full" ..."""
import json, sys
from collections import Counter
import check
from check import Reader, check_photo
from e2e import score
from realset import split_of

truth = {k: v for k, v in json.load(open("testset.json")).items() if not k.startswith("_")}
import os
reader = Reader(hand_head=os.environ.get("HEAD"))
for combo in sys.argv[1:]:
    check.VIEWS = combo.split(",")
    c = {s: Counter() for s in ("train", "calib", "test")}
    for f, tables in truth.items():
        score(f, tables, check_photo(f"../{f}", reader)[0], c[split_of(f)], verbose=True)
    parts = []
    for s, cc in c.items():
        n = sum(v for k, v in cc.items() if isinstance(k, tuple))
        rv = sum(v for k, v in cc.items() if isinstance(k, tuple) and k[1] == "review")
        parts.append(f"{s}: {n - rv}/{n} auto, {cc[('child right', 'wrong')] + cc[('child wrong', 'right')]} wrong marks")
    print(f"VIEWS={combo:28} " + " | ".join(parts), flush=True)
