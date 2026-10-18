"""End-to-end score: run the full checker on each photo and compare every verdict with the hand transcription.

usage: python e2e.py            (all photos; held-out test photos are reported separately)
"""
import json
from collections import Counter
from check import Reader, check_photo, evaluate, fmt
from realset import TEST_PHOTOS



def score(fname, tables, results, c, verbose=True):
    """Add the checker's verdicts for one photo to Counter c, compared with the transcription."""
    got_tables = sorted({r["table"] for r in results})
    if len(got_tables) != len(tables):
        if verbose:
            print(f"{fname}: checker found {len(got_tables)} tables, expected {len(tables)}")
        c["missed"] += sum(len(t["answers"]) for t in tables)
        return
    for t_idx, gt in zip(got_tables, tables):
        res = [r for r in results if r["table"] == t_idx]
        for r, nums, written in zip(res, gt["numbers"], gt["answers"]):
            if written == "?" or "?" in nums:
                continue
            correct = fmt(evaluate([str(n) for n in nums]))
            child = "child right" if written == correct else "child wrong"
            c[(child, r["status"])] += 1
            c["printed ok" if r["numbers"] == [str(n) for n in nums] else "printed misread"] += 1
            if verbose and (child, r["status"]) in [("child right", "wrong"), ("child wrong", "right")]:
                print(f"  !! {fname[-14:-5]} T{t_idx} Q{r['q']}: wrote {written}, correct {correct}, "
                      f"read {r['given']} {r['numbers']} -> {r['status']}")



if __name__ == "__main__":
    truth = json.load(open("testset.json"))
    reader = Reader()
    totals = {"train": Counter(), "test": Counter()}
    for fname, tables in truth.items():
        if not fname.startswith("_"):
            score(fname, tables, check_photo(f"../{fname}", reader)[0],
                  totals["test" if fname in TEST_PHOTOS else "train"])

    for split, c in totals.items():
        n = sum(v for k, v in c.items() if isinstance(k, tuple))
        review = sum(v for k, v in c.items() if isinstance(k, tuple) and k[1] == "review")
        bad = c[("child right", "wrong")] + c[("child wrong", "right")]
        print(f"\n{split.upper()} photos: {n} answers")
        for child in ["child right", "child wrong"]:
            print(f"  {child:12}: marked right {c[(child, 'right')]:3}  marked wrong {c[(child, 'wrong')]:3}  "
                  f"review {c[(child, 'review')]:3}  blank {c[(child, 'blank')]:2}")
        print(f"  checked automatically: {n - review}/{n} ({(n - review) / max(n, 1):.0%})   "
              f"mis-marked: {bad}   questions with a printed-number misread: {c['printed misread']}")
