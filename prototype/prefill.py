"""Pre-fill a transcription from the checker's readings, plus straightened table images to verify it.

usage: python prefill.py <photo> <out_dir>
  -> <out_dir>/<stem>_T<n>.jpg (straightened tables with the checker's readings written under each column)
     prints a JSON fragment {"numbers": [...], "answers": [...]} per table to correct by hand
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from check import Reader, check_table, worksheet_tables

path, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
reader = Reader()
tables = []
for ti, (t, xs) in enumerate(worksheet_tables(cv2.imread(path), reader)):
    res = check_table(t, xs, reader)
    vis = cv2.copyMakeBorder(t, 0, 230, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for r in res:
        x0, _, x1, y1 = r["box"]
        lines = r["numbers"] + ["=" + (r["given"] or "_")]
        for k, txt in enumerate(lines[-11:]):
            cv2.putText(vis, txt, (x0 + 6, t.shape[0] + 18 + k * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 0, 0), 1)
    cv2.imwrite(str(out / f"{Path(path).stem[-14:].strip()}_T{ti + 1}.jpg"), vis)
    tables.append({"numbers": [[int(n) if n.lstrip("-").isdigit() else n for n in r["numbers"]] for r in res],
                   "answers": [r["given"] for r in res]})
print(json.dumps({Path(path).name: tables}))
