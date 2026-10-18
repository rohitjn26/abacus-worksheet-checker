"""Export label-column cells (ANS / A / B / C / blank) as raw crops, labelled by our reader (100% on held-out
label cells vs Apple Vision earlier). PaddleOCR list format: 'path<TAB>label<TAB>label'.

usage: python export_label_cells.py <out_dir>
"""
import glob
import sys
from pathlib import Path
import cv2
from cells import label_cells, table_rows
from check import Reader, worksheet_tables
from norm import normalize
from realset import TEST_PHOTOS

out = Path(sys.argv[1])
reader = Reader(hand="own")
lists = {"train": [], "test": []}
for f in sorted(glob.glob("../WhatsApp*.jpeg")):
    split = "test" if Path(f).name in TEST_PHOTOS else "train"
    for t, xs in worksheet_tables(cv2.imread(f), reader):
        col_ys, _, lines = table_rows(t, xs)
        for _, cell in label_cells(t, xs, col_ys, lines):
            text, conf = reader([cell], printed=True)[0]
            if normalize(cell, printed=True).max() == 0:
                text = ""
            elif text not in ("ANS", "A", "B", "C") or conf < 0.9:
                continue
            name = f"{split}/label_{len(lists[split]):05d}.png"
            (out / split).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / name), cell)
            lists[split].append(f"{name}\t{text}\tlabel")
for split, rows in lists.items():
    (out / f"{split}_labels.txt").write_text("\n".join(rows) + "\n")
    print(split, len(rows), "label cells")
