"""Split a straightened table into per-question cells: printed number cells + handwritten answer cell.

Rows are found per column strip (a narrow strip is barely affected by page curl / camera tilt).
Missing row lines are filled in from the regular row pitch. The number rows are then found by
content: walking up from the ANS row, a row is a number row while most of its cells read as numbers.
"""
import re
import cv2
import numpy as np
from grid import angled_open, binarize, line_masks

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


def keep_pen_on_lines(gray, mask, near=3, far=7):
    """Remove from a grid-line mask the places where a pen stroke crosses or leaves the line, so erasing
    the line does not erase the stroke (e.g. the top of a 9 drawn on the ANS line). A stretch of line is
    'written on' where ink continues a few pixels away from it (above / below a horizontal line, left /
    right of a vertical one) - a digit merely resting near the line does not count."""
    ink = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 12)
    band = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    off = (ink > 0) & (band == 0)
    away = np.zeros_like(off)
    for d in range(near, far + 1):  # ink d pixels above / below / left / right of a line pixel
        away |= np.roll(off, d, 0) | np.roll(off, -d, 0) | np.roll(off, d, 1) | np.roll(off, -d, 1)
    keep = (band > 0) & away & (cv2.dilate(off.astype(np.uint8), np.ones((near + 2, near + 2), np.uint8)) > 0)
    return np.where(keep, 0, mask).astype(np.uint8)


def printed_bottom(table, lines, x0, x1, y0, y1):
    """Lowest row of the printed number's ink inside the number cell (y0, y1), in table coordinates.
    Blobs touching the cell's bottom line are the child's writing reaching up, not the printed number."""
    cell = clean_crop(table, lines, x0, x1, y0, y1)
    bw = cv2.adaptiveThreshold(cell, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)
    n, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    h = y1 - y0
    bottoms = [stats[i, 1] + stats[i, 3] for i in range(1, n)
               if stats[i, 3] > 0.25 * h and stats[i, 1] + stats[i, 3] < h - 3]
    return y0 + max(bottoms) if bottoms else y0 + h // 2


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
    # full: from just below the last printed number down to the bottom of the second ANS row - all the room
    # the child's writing uses - with grid lines erased only where no pen touches them.
    top = printed_bottom(table, lines, x0, x1, ys[j - 1], ys[j]) + 2 if j > 0 else max(ys[j] - p // 2, 0)
    bottom = min(below2 + int(p * 0.15), table.shape[0])
    gray = cv2.cvtColor(table[top:bottom, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = keep_pen_on_lines(gray, cv2.dilate(lines[top:bottom, x0:x1], np.ones((3, 3), np.uint8)))
    full = cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)
    return {"tight": tight, "rows": rows, "above": above, "full": full}


def extend_to_edges(ys, pitch, height):
    """Row lines lost at the very top / bottom of the crop (border cut off or faint): if most of a row's
    height of table remains beyond the last line found, add lines at the normal row spacing."""
    if not ys:
        return ys
    ys = list(ys)
    while height - 1 - ys[-1] > 0.7 * pitch:
        ys.append(min(int(round(ys[-1] + pitch)), height - 1))
    while ys[0] > 0.7 * pitch:
        ys.insert(0, max(int(round(ys[0] - pitch)), 0))
    return ys


def clean_crop(table, lines_mask, x0, x1, y0, y1, pad_y=0, pad_bottom=None):
    """Grey crop of a cell with its grid lines inpainted from the surrounding paper."""
    H = table.shape[0]
    a, b = max(y0 - pad_y, 0), min(y1 + (pad_y if pad_bottom is None else pad_bottom), H)
    gray = cv2.cvtColor(table[a:b, x0:x1], cv2.COLOR_BGR2GRAY)
    mask = cv2.dilate(lines_mask[a:b, x0:x1], np.ones((3, 3), np.uint8))
    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA)


def table_rows(table, xs):
    """Row lines per column, the row pitch, and a mask of the grid lines (for painting them out)."""
    gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    col_w = float(np.median(np.diff(xs)))
    # Row lines may tilt (page curl); strokes longer than 0.4 column width are grid lines, not digits.
    horiz = angled_open(binarize(gray), max(int(col_w * 0.4), 10), range(-15, 16, 3), vertical=False)
    col_ys = [strip_rows(horiz, xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
    gaps = [d for ys in col_ys[1:] for d in np.diff(ys) if d > 8]
    if not gaps:
        raise ValueError("no row lines found")
    pitch = float(np.median(gaps))
    col_ys = [extend_to_edges(fill_gaps(ys, pitch), pitch, table.shape[0]) for ys in col_ys]

    # Mask only true grid lines for painting out: longer than any digit stroke (a '1' is < 1 row tall).
    _, v_long = line_masks(gray, vlen=pitch * 1.6)
    lines = cv2.dilate(horiz | v_long, np.ones((3, 3), np.uint8))
    return col_ys, pitch, lines


def label_cells(table, xs, col_ys, lines):
    """(top y, crop) for each cell of the label column (A / blank / ANS / ANS ...)."""
    ys = col_ys[0]
    return [(a, clean_crop(table, lines, xs[0], xs[1], a, b)) for a, b in zip(ys, ys[1:])]


def split_table(table, xs, read, is_ans):
    """read(cell) -> text of a printed cell; is_ans(cell) -> True if a label cell says 'ANS'."""
    col_ys, pitch, lines = table_rows(table, xs)

    # Label column: the first 'ANS' cell from the top.
    ans_y = next((a for a, cell in label_cells(table, xs, col_ys, lines) if is_ans(cell)), None)
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
    gaps = [False for _ in cols]  # a number row where this column's cell came out blank (glare, fading...)
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
        for i, (n, (cell, t)) in enumerate(zip(numbers, texts)):
            if cell is not None and t:
                n.insert(0, (cell, t))
            else:
                gaps[i] = True

    return [{"numbers": n, "missing_number": g, "answer": c["answer"], "answer_crops": c["answer_crops"],
             "answer_box": c["answer_box"]}
            for n, g, c in zip(numbers, gaps, cols)]
