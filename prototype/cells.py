"""Split a straightened table into per-question cells: printed number cells + handwritten answer cell.

Rows are found per column strip (a narrow strip is barely affected by page curl / camera tilt).
Missing row lines are filled in from the regular row pitch. The number rows are then found by
content: walking up from the ANS row, a row is a number row while most of its cells read as numbers.
"""
import re
import cv2
import numpy as np
from grid import angled_open, binarize, line_masks, ocr

NUMBER = re.compile(r"^[x×*÷/+\-]?-?\d+(\.\d+)?$")


def strip_rows(horiz, x0, x1):
    """y of each row line crossing a column strip, taken at the strip centre (lines may be tilted)."""
    pad = max((x1 - x0) // 10, 3)
    strip = horiz[:, x0 + pad : x1 - pad]
    n, labels, stats, cents = cv2.connectedComponentsWithStats(strip)
    ys = sorted(int(cents[i][1]) for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] >= 0.6 * strip.shape[1])
    return [y for i, y in enumerate(ys) if i == 0 or y - ys[i - 1] > 8]


def fill_gaps(ys, pitch):
    """Insert evenly spaced lines where a gap is ~k times the normal row pitch."""
    out = ys[:1]
    for y in ys[1:]:
        k = round((y - out[-1]) / pitch)
        if k >= 2 and abs((y - out[-1]) / k - pitch) < pitch * 0.2:
            out += [round(out[-1] + (y - out[-1]) * j / k) for j in range(1, k)]
        out.append(y)
    return out


def blank_printed(gray, y_cut):
    """Paint out ink blobs that sit wholly above y_cut (the printed number of the last number row),
    keeping handwriting that reaches up into that row from the ANS row below."""
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.dilate(bw, np.ones((3, 3), np.uint8)))
    out = gray.copy()
    paper = int(np.percentile(gray, 80))
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] < y_cut:
            out[labels == i] = paper
    return out


def answer_crops(table, lines, x0, x1, ys, j, pitch):
    """Answer crops from tight to wide; wider ones catch digits spilling out of the first ANS row.

    tight:  first ANS row (+ a bit below for tails)
    rows:   both ANS rows
    above:  both ANS rows + the lower part of the last number row (its printed number blanked out)
    """
    p = int(pitch)
    below2 = ys[j + 2] if j + 2 < len(ys) else ys[j + 1] + p
    tight = clean_crop(table, lines, x0, x1, ys[j], ys[j + 1], pad_y=int(p * 0.15), pad_bottom=int(p * 0.7))
    rows = clean_crop(table, lines, x0, x1, ys[j], below2, pad_y=int(p * 0.15), pad_bottom=int(p * 0.15))
    up = int(p * 0.6)
    above = clean_crop(table, lines, x0, x1, ys[j], below2, pad_y=up, pad_bottom=int(p * 0.15))
    above = blank_printed(above, y_cut=min(up, ys[j]) - 1)
    return {"tight": tight, "rows": rows, "above": above}


def clean_crop(table, lines_mask, x0, x1, y0, y1, pad_y=0, pad_bottom=None):
    """Grey crop of a cell with its grid lines inpainted from the surrounding paper."""
    H = table.shape[0]
    a, b = max(y0 - pad_y, 0), min(y1 + (pad_y if pad_bottom is None else pad_bottom), H)
    gray = cv2.cvtColor(table[a:b, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = cv2.dilate(lines_mask[a:b, x0:x1], np.ones((3, 3), np.uint8))
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


def read_printed(cell):
    """OCR one printed cell. Returns normalised text ('' if empty)."""
    big = cv2.resize(cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR), None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    big = cv2.copyMakeBorder(big, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    text = "".join(t for t, _, _ in ocr(big)).replace(" ", "")
    return text.translate(str.maketrans({"−": "-", "–": "-", "—": "-", "X": "×", "x": "×", "*": "×"}))


def split_table(table, xs, read=None):
    """read(cell) -> text for a printed cell; defaults to Apple Vision OCR."""
    read = read or read_printed
    gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    col_w = float(np.median(np.diff(xs)))
    # Row lines may tilt (page curl); strokes longer than 0.4 column width are grid lines, not digits.
    horiz = angled_open(binarize(gray), max(int(col_w * 0.4), 10), range(-15, 16, 3), vertical=False)
    col_ys = [strip_rows(horiz, xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
    gaps = [d for ys in col_ys[1:] for d in np.diff(ys) if d > 8]
    if not gaps:
        raise ValueError("no row lines found")
    pitch = float(np.median(gaps))
    col_ys = [fill_gaps(ys, pitch) for ys in col_ys]

    # Mask only true grid lines for painting out: longer than any digit stroke (a '1' is < 1 row tall).
    _, v_long = line_masks(gray, vlen=pitch * 1.6)
    lines = cv2.dilate(horiz | v_long, np.ones((3, 3), np.uint8))

    # Label column: find the first 'ANS' cell.
    label_ys = col_ys[0]
    ans_y = None
    for a, b in zip(label_ys, label_ys[1:]):
        if "ANS" in " ".join(t for t, _, _ in ocr(table[a:b, xs[0] : xs[1]])).upper():
            ans_y = a
            break
    if ans_y is None:
        raise ValueError("no ANS row found in label column")

    # Per question column: the ANS row, and the rows above it (nearest first).
    cols = []
    prev = ans_y
    for i in range(1, len(xs) - 1):
        ys = col_ys[i]
        j = min(range(len(ys)), key=lambda k: abs(ys[k] - prev))
        if abs(ys[j] - prev) > pitch * 0.5 or j + 1 >= len(ys):
            raise ValueError(f"column {i}: cannot locate ANS row")
        prev = ys[j]
        above = [(ys[k - 1], ys[k]) for k in range(j, 0, -1)]
        crops = answer_crops(table, lines, xs[i], xs[i + 1], ys, j, pitch)
        cols.append({"x": (xs[i], xs[i + 1]), "above": above, "answer": crops["tight"], "answer_crops": crops,
                     "answer_box": (xs[i], ys[j], xs[i + 1], ys[j + 1])})

    # Walk upward from ANS; keep rows where most columns read as numbers. Mostly-blank rows (ragged
    # columns) are skipped; anything else (header row I, II, III...) ends the number block.
    numbers = [[] for _ in cols]
    for r in range(max(len(c["above"]) for c in cols)):
        texts = []
        for c in cols:
            if r < len(c["above"]):
                a, b = c["above"][r]
                cell = clean_crop(table, lines, *c["x"], a, b)
                texts.append((cell, read(cell)))
            else:
                texts.append((None, ""))
        n_num = sum(bool(NUMBER.match(t)) for _, t in texts)
        n_blank = sum(t == "" for _, t in texts)
        if n_num < len(cols) / 2:
            if n_blank > len(cols) / 2:
                continue
            break
        for n, (cell, t) in zip(numbers, texts):
            if cell is not None and t:
                n.insert(0, (cell, t))

    return [{"numbers": n, "answer": c["answer"], "answer_crops": c["answer_crops"], "answer_box": c["answer_box"]}
            for n, c in zip(numbers, cols)]
