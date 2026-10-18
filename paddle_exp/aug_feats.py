"""Frozen PaddleOCR features (ONNX) for the augmented cells: aug_feats.npz (x, y, kind)."""
import sys
import cv2
import numpy as np
sys.path.insert(0, "../prototype")
from hand_reader import HandReader  # noqa: E402

fx = HandReader(onnx_path="../prototype/ppocrv5_mobile_frozen.onnx", head_path="../prototype/hand_head.pt")
rows = [l.split("\t") for l in open("cells/aug.txt").read().splitlines() if l]
feats = []
for i in range(0, len(rows), 128):
    feats.append(fx.features([cv2.imread(f"cells/{r[0]}", cv2.IMREAD_GRAYSCALE) for r in rows[i : i + 128]]).astype(np.float16))
np.savez_compressed("aug_feats.npz", x=np.concatenate(feats), y=[r[1] for r in rows], kind=[r[2] for r in rows])
print("aug features", sum(len(f) for f in feats))
