"""Score model variants end to end on every split (train / calibration / held-out test).

usage: python compare_quant.py reader.onnx:head.onnx:paddle.onnx [...]
"""
import json
import sys
from collections import Counter
from check import Reader, check_photo
from e2e import score
from realset import split_of

truth = {k: v for k, v in json.load(open("testset.json")).items() if not k.startswith("_")}
for combo in sys.argv[1:]:
    reader_file, head_file, paddle_file = combo.split(":")
    r = Reader(weights=reader_file, hand_head=head_file, paddle=paddle_file)
    c = {s: Counter() for s in ("train", "calib", "test")}
    for f, tables in truth.items():
        score(f, tables, check_photo(f"../{f}", r)[0], c[split_of(f)], verbose=False)
    parts = []
    for s, cc in c.items():
        n = sum(v for k, v in cc.items() if isinstance(k, tuple))
        rv = sum(v for k, v in cc.items() if isinstance(k, tuple) and k[1] == "review")
        bad = cc[("child right", "wrong")] + cc[("child wrong", "right")]
        parts.append(f"{s}: {n - rv}/{n} auto, {bad} wrong marks")
    print(f"{combo}\n    " + " | ".join(parts), flush=True)
