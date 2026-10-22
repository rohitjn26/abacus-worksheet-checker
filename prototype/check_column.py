"""Check a close-up photo of ONE column (header, printed numbers, then the ANS row(s)), without the table's
label column. Same models and right / wrong / review rules as check.py.

usage: python check_column.py <image> [...]   -> prints the verdicts, writes results/<name>_checked.jpg
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import ctc
from cells import NUMBER, clean_crop
from check import ABSENT_MAX, Reader, evaluate, fmt, judge
from grid import angled_open, binarize


def column_cells(img):
    """Row bands of the column between its two long vertical lines: [(y0, y1, crop)] top to bottom."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bw = binarize(gray)
    horiz = angled_open(bw, int(w * 0.35), range(-6, 7, 2), vertical=False)
    vert = angled_open(bw, int(h * 0.5), range(-6, 7, 2), vertical=True)
    # column edges: strongest vertical lines left and right of the middle
    prof = vert.sum(0)
    left = int(np.argmax(prof[: w // 2])) if prof[: w // 2].max() > 0 else 0
    right = w // 2 + int(np.argmax(prof[w // 2 :])) if prof[w // 2 :].max() > 0 else w - 1
    rows = horiz[:, left:right].sum(1) > 0.5 * (right - left) * 255
    ys, start = [], None
    for y, on in enumerate(np.append(rows, False)):
        if on and start is None:
            start = y
        elif not on and start is not None:
            ys.append((start + y - 1) // 2)
            start = None
    ys = [0] + [y for y in ys if 4 < y < h - 4] + [h - 1]
    lines = cv2.dilate(horiz | vert, np.ones((3, 3), np.uint8))
    x0, x1 = left + 3, right - 3
    return [(a, b, clean_crop(img, lines, x0, x1, a, b)) for a, b in zip(ys, ys[1:]) if b - a > 12], (x0, x1)


def check_column(img, reader):
    cells, (x0, x1) = column_cells(img)
    # Only full-height rows count (bits cut off at the photo's top / bottom edge are dropped).
    pitch = float(np.median([b - a for a, b, _ in cells]))
    cells = [c for c in cells if c[1] - c[0] > 0.7 * pitch]
    reads = [(a, b, c, *reader([c], printed=True)[0]) for a, b, c in cells]
    # Layout: header (I, II, ...), the printed numbers, then the two ANS rows at the bottom. The child's
    # answer can read like a number, so the ANS rows are found by position, not by content.
    head = next((i for i, r in enumerate(reads) if r[3] and set(r[3]) <= set("IVX")), -1)
    first, last = head + 1, len(reads) - 3
    if last < first or not all(NUMBER.match(reads[i][3]) for i in range(first, last + 1)):
        return None
    nums = [reads[i][3] for i in range(first, last + 1)]
    p_conf = min(reads[i][4] for i in range(first, last + 1))
    correct = evaluate(nums)
    ans_rows = reads[last + 1 :]
    crops = {}
    if ans_rows:
        crops["row 1"] = ans_rows[0][2]
        if len(ans_rows) > 1:
            crops["row 2"] = ans_rows[1][2]
            a, b = ans_rows[0][0], ans_rows[1][1]
            crops["both rows"] = cv2.cvtColor(img[a:b, x0:x1], cv2.COLOR_BGR2GRAY)
    lps, abc = {}, None
    for k, c in crops.items():
        lps[k], abc = reader.answer_log_probs([c])
    verdicts = {k: judge(lp, correct, abc) for k, lp in lps.items()} if correct is not None else {}
    # the child may have written in either ANS row (or crossed one out): use the clearest reading
    inked = {k: v for k, v in verdicts.items() if v[0] != "blank"}
    if correct is None or p_conf < 0.9:
        status, given, where = "review", "", "printed numbers unclear"
    elif not inked:
        status, given, where = "blank", "", ""
    else:
        where = max(inked, key=lambda k: inked[k][2])
        status, given, _ = inked[where]
        p_correct = max(float(np.exp(ctc.string_logprob(lp, [fmt(correct)], abc)[0])) for lp in lps.values())
        if status == "review" and p_correct < ABSENT_MAX:
            status = "wrong"
    return {"numbers": nums, "correct": fmt(correct) if correct is not None else "?", "given": given,
            "status": status, "where": where, "readings": {k: (v[1], round(v[2], 2)) for k, v in verdicts.items()}}


if __name__ == "__main__":
    reader = Reader()
    Path("results").mkdir(exist_ok=True)
    for path in sys.argv[1:]:
        img = cv2.imread(path)
        r = check_column(img, reader)
        if r is None:
            print(f"{Path(path).name}: no printed numbers found")
            continue
        print(f"{Path(path).name}: {' + '.join(r['numbers']).replace('+ -', '- ')} = {r['correct']} | child: "
              f"{r['given'] or '-'} ({r['where']}) -> {r['status'].upper()}   readings {r['readings']}")
        color = {"right": (40, 160, 40), "wrong": (30, 30, 220), "review": (0, 140, 255), "blank": (150, 150, 150)}[r["status"]]
        label = {"right": "OK", "wrong": f"X {r['correct']}", "review": f"? = {r['correct']}", "blank": "-"}[r["status"]]
        out = cv2.copyMakeBorder(img, 0, 40, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(out, label, (6, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        cv2.imwrite(f"results/{Path(path).stem}_checked.jpg", out)
