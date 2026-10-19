"""Frozen-PaddleOCR features (ONNX) for every labelled real cell, in the layout train_hand_head.py reads:
train_x/train_y (training photos), test_x/test_y/test_kind (held-out photos), synth_x/synth_y (copied from
an earlier features file).

usage: python build_hand_feats.py <old_feats.npz with synth_x> <out.npz>
"""
import sys
import numpy as np
from hand_reader import HandReader
from realset import TEST_PHOTOS, load

fx = HandReader()
data = load()
old = np.load(sys.argv[1])
out = {"synth_x": old["synth_x"], "synth_y": old["synth_y"]}
for split, keep in [("train", lambda p: p not in TEST_PHOTOS), ("test", lambda p: p in TEST_PHOTOS)]:
    rows = [d for d in data if keep(d[3])]
    crops = [d[5] for d in rows]
    out[f"{split}_x"] = np.concatenate([fx.features(crops[i : i + 128]) for i in range(0, len(crops), 128)]).astype(np.float16)
    out[f"{split}_y"] = np.array([d[1] for d in rows])
    out[f"{split}_kind"] = np.array([d[2] for d in rows])
    print(split, len(rows), "cells:", (out[f"{split}_kind"] == "hand").sum(), "handwritten", flush=True)
np.savez_compressed(sys.argv[2], **out)
