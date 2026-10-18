"""Export the labelled real cells as plain image files + label lists (PaddleOCR format: 'path<TAB>label').

usage: python export_cells.py <out_dir>    -> <out_dir>/{train,test}/*.png, train.txt, test.txt
"""
import sys
from pathlib import Path
import cv2
from realset import TEST_PHOTOS, load

out = Path(sys.argv[1])
lists = {"train": [], "test": []}
for i, (_, label, kind, photo, tag, raw) in enumerate(load()):
    split = "test" if photo in TEST_PHOTOS else "train"
    (out / split).mkdir(parents=True, exist_ok=True)
    name = f"{split}/{kind}_{i:05d}.png"
    cv2.imwrite(str(out / name), raw)
    lists[split].append(f"{name}\t{label}\t{kind}")
for split, rows in lists.items():
    (out / f"{split}.txt").write_text("\n".join(rows) + "\n")
    print(split, len(rows))
