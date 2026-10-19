"""Check every photo in a folder: verdict counts, retake requests, and marked sheets.

usage: python run_folder.py <folder> [out_dir=results_new]
"""
import glob
import sys
from collections import Counter
from pathlib import Path
import cv2
from check import Reader, check_photo
from quality import retake_reasons

folder, out = sys.argv[1], Path(sys.argv[2] if len(sys.argv) > 2 else "results_new")
out.mkdir(exist_ok=True)
reader, total = Reader(), Counter()
for f in sorted(glob.glob(f"{folder}/*.jp*g")):
    report, img = {}, cv2.imread(f)
    res, sheet = check_photo(img, reader, report)
    reasons = [k for k, _ in retake_reasons(img, res, report)]
    c = Counter(x["status"] for x in res)
    total.update(c)
    print(f"{Path(f).stem[-14:]:>14}: tables {len({x['table'] for x in res})}, right {c['right']:2}, wrong {c['wrong']:2}, "
          f"review {c['review']:2}, blank {c['blank']:2}" + (f"   RETAKE: {', '.join(reasons)}" if reasons else ""))
    cv2.imwrite(str(out / f"{Path(f).stem}_checked.jpg"), sheet)
print("TOTAL", dict(total))
