"""Check a worksheet photo end to end.

usage: python check.py <photo> [<photo> ...]   ->  prints results, writes results/<name>_checked.jpg
"""
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
import cv2
import numpy as np
import torch
from cells import split_table
from grid import find_tables, fix_orientation, grid_lines
from model import CRNN, decode
from norm import normalize
from model import string_logprob
from policy import decide

PRINTED_MIN_CONF = 0.90   # below this, a printed number is not trusted -> teacher review (see calibrate.py)
CELL = re.compile(r"^([×÷+]?)(-?\d+(?:\.\d+)?)$")


class Reader:
    def __init__(self, weights="reader_ft.pt"):
        self.model = CRNN()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.eval()

    def log_probs(self, crops, printed=False):
        x = torch.from_numpy(np.stack([normalize(c, printed) for c in crops]))[:, None]
        with torch.no_grad():
            return self.model(x)

    def __call__(self, crops, printed=False):
        return decode(self.log_probs(crops, printed)) if crops else []


def evaluate(cells):
    """Top-to-bottom abacus evaluation. cells: ['5', '-3', '×4', '÷2', ...] -> Decimal, or None if unparseable."""
    total = None
    for c in cells:
        m = CELL.match(c)
        if not m:
            return None
        op, v = m.group(1), Decimal(m.group(2))
        if total is None:
            total = v
        elif op == "×":
            total *= v
        elif op == "÷":
            if v == 0:
                return None
            total /= v
        else:
            total += v  # '+5', '5' and '-5' are all "add this signed number"
    return total


def as_number(text):
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def fmt(d):
    return format(d.normalize(), "f") if d is not None else "?"


ABSENT_MAX = 1e-5  # if even the best crop gives "child wrote exactly the correct answer" less than this
                   # probability, the answer is wrong (wrong digit count, a clear digit not in the answer...)
WIDE_CROPS = True  # if the tight crop is unclear, also try both ANS rows / both rows + the strip above


def judge(lp, correct):
    """Read one answer from its log-probs and decide right / wrong / review against the correct answer."""
    (ans, conf), = decode(lp)
    given = as_number(ans)
    if not ans:
        return "blank", ans, conf
    if given is None:
        return "review", ans, conf
    read = fmt(given)
    alt = (string_logprob(lp, [fmt(correct)]) - string_logprob(lp, [ans])).exp().item()
    return decide(read, conf, fmt(correct), alt), ans, conf


def check_table(table, reader):
    table, xs = grid_lines(table)
    if len(xs) < 4:
        raise ValueError("no column grid")
    qs = split_table(table, xs, read=lambda cell: reader([cell], printed=True)[0][0])
    kinds = ["tight", "rows", "above"] if WIDE_CROPS else ["tight"]
    lps = {k: reader.log_probs([q["answer_crops"][k] for q in qs]) for k in kinds}
    results = []
    for i, q in enumerate(qs):
        nums = [t for _, t in q["numbers"]]
        p_conf = min((conf for _, conf in reader([c for c, _ in q["numbers"]], printed=True)), default=0)
        correct = evaluate(nums)
        r = {"q": i + 1, "numbers": nums, "correct": correct, "box": q["answer_box"]}
        (ans, conf), = decode(lps["tight"][:, i : i + 1])
        r["given"], r["conf"] = ans, conf
        if correct is None or p_conf < PRINTED_MIN_CONF:
            r["status"], r["why"] = ("blank", "") if not ans else ("review", "couldn't read the printed numbers")
            if correct is None:
                r["why"] = r["why"] or "couldn't read the printed numbers"
            results.append(r)
            continue
        verdicts = {k: judge(lps[k][:, i : i + 1], correct) for k in kinds}
        status, ans, conf = verdicts["tight"]
        p_correct = max(string_logprob(lps[k][:, i : i + 1], [fmt(correct)]).exp().item() for k in kinds)
        if status == "review" and ans and p_correct < ABSENT_MAX:
            status = "wrong"  # whatever exactly the child wrote, it is not the correct answer
        if status == "review" and len(kinds) > 1:
            # Digits may spill out of the first ANS row: use the crop the reader is most sure about.
            best = max(kinds, key=lambda k: verdicts[k][2])
            status, ans, conf = verdicts[best]
            r["crop"] = best
        r["status"], r["given"], r["conf"] = status, ans, conf
        if status == "review":
            r["why"] = f"handwriting unclear (best guess {ans})"
        results.append(r)
    return results


COLORS = {"right": (40, 160, 40), "wrong": (30, 30, 220), "review": (0, 140, 255), "blank": (150, 150, 150)}


def draw(table, results):
    vis = cv2.copyMakeBorder(table, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))  # room for labels
    for r in results:
        x0, y0, x1, y1 = r["box"]
        c = COLORS[r["status"]]
        cv2.rectangle(vis, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), c, 3)
        expected = fmt(r["correct"]) if r["correct"] is not None else "?"
        if r.get("why", "").startswith("couldn't read the printed"):
            expected += "*"  # computed from printed numbers the reader was unsure of
        label = {"right": "OK", "wrong": f"X {expected}", "review": f"? = {expected}",
                 "blank": f"- = {expected}"}[r["status"]]
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(vis, (x0 + 2, y1 + 4), (x0 + 8 + tw, y1 + 10 + th), (255, 255, 255), -1)  # readable on grey
        cv2.putText(vis, label, (x0 + 5, y1 + 7 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.65, c, 2, cv2.LINE_AA)
    vis = cv2.copyMakeBorder(vis, 0, 60, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    legend = "OK = right   X n = wrong (n = correct answer)   ? = n : please check, expected answer n"
    if any(r.get("why", "").startswith("couldn't read the printed") for r in results):
        legend += "   * = printed numbers unclear, verify the sum"
    cv2.putText(vis, legend, (8, vis.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1, cv2.LINE_AA)
    return vis


SCAN_LOOK = False  # experiment: even out shadows/lighting over the whole photo first, like a scanner app


def scan_look(img):
    """Gentle 'scan': divide by the smooth background so shadows and gradients vanish; grey levels are kept."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), 41).astype(np.float32)
    out = img.astype(np.float32) / np.maximum(bg[..., None], 1) * 235
    return np.clip(out, 0, 255).astype(np.uint8)


def check_photo(path, reader):
    img = cv2.imread(str(path))
    if SCAN_LOOK:
        img = scan_look(img)
    img = fix_orientation(img)
    tables = find_tables(img)
    all_results, panels = [], []
    for ti, t in enumerate(tables):
        try:
            res = check_table(t, reader)
        except ValueError:  # not a worksheet table (e.g. book edge), or no ANS row found
            continue
        for r in res:
            r["table"] = ti + 1
        all_results += res
        panels.append(draw(t, res))
    if not panels:
        return [], img  # no worksheet table found: return the photo unmarked
    w = max(p.shape[1] for p in panels)
    sheet = np.vstack([cv2.copyMakeBorder(p, 10, 10, 0, w - p.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
                       for p in panels])
    return all_results, sheet


def main():
    reader = Reader()
    out = Path("results")
    out.mkdir(exist_ok=True)
    for path in sys.argv[1:]:
        results, sheet = check_photo(path, reader)
        if not results:
            print(f"\n== {Path(path).name}: no worksheet table found - please retake the photo ==")
            continue
        right = sum(r["status"] == "right" for r in results)
        print(f"\n== {Path(path).name}: {right}/{len(results)} right ==")
        for r in results:
            if r["status"] != "right":
                expr = " ".join(r["numbers"])
                extra = f"  ({r['why']})" if "why" in r else ""
                print(f"  T{r['table']} Q{r['q']:>2}: {r['status'].upper():6} {expr} = {fmt(r['correct'])}, "
                      f"child wrote {r['given'] or '(blank)'}{extra}")
        cv2.imwrite(str(out / f"{Path(path).stem}_checked.jpg"), sheet)


if __name__ == "__main__":
    main()
