"""Dump every verdict of the checker on the transcribed photos, to compare two versions of the code.

usage: python parity_dump.py <photos_root> <out.json>
"""
import json
import sys
from check import Reader, check_photo

root, out = sys.argv[1], sys.argv[2]
truth = {k: v for k, v in json.load(open("testset.json")).items() if not k.startswith("_")}
reader, rows = Reader(), []
for f in truth:
    report = {}
    for r in check_photo(f"{root}/{f}", reader, report)[0]:
        rows.append([f, r["table"], r["q"], r["status"], r["given"], r["numbers"], str(r["correct"])])
    rows.append([f, "unread_tables", report.get("unread_tables", 0)])
json.dump(rows, open(out, "w"))
print(len(rows), "rows")
