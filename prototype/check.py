"""Check a worksheet photo end to end.

usage: python check.py <photo> [<photo> ...]   ->  prints results, writes results/<name>_checked.jpg
"""
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
import cv2
import numpy as np
import ctc
from cells import label_cells, split_table, table_rows
from grid import find_tables, fix_orientation, grid_lines
from norm import ALPHABET, normalize
from policy import decide
from runtime import HAND_ALPHABET, Models
from quality import retake_reasons

PRINTED_MIN_CONF = 0.90   # below this, a printed number is not trusted -> teacher review (see calibrate.py)
CELL = re.compile(r"^([×÷+]?)(-?\d+(?:\.\d+)?)$")


HAND_READER = "paddle"  # "paddle": PaddleOCR (frozen) + trained head for answers; "own": our model
READER_WEIGHTS = "reader_v4_int8.onnx"  # our model (printed numbers, headers, ANS); 8-bit, same results as reader_v4
HAND_HEAD = "hand_head_v2.pt"     # head on frozen PaddleOCR for handwritten answers
PADDLE_ONNX = "ppocrv5_mobile_frozen_fp16w.onnx"  # frozen PaddleOCR layers; weights stored 16-bit, same results


class Reader:
    """Reads cells with the ONNX models (runtime.py). backend: None = onnxruntime; the Android app passes
    its Kotlin bridge."""

    def __init__(self, weights=None, hand=None, hand_head=None, backend=None, paddle=None):
        self.models = Models(reader=weights or READER_WEIGHTS, hand_head=hand_head or HAND_HEAD,
                             paddle=paddle or PADDLE_ONNX, backend=backend)
        self.hand = (hand or HAND_READER) == "paddle"

    def answer_log_probs(self, crops):
        """(log-probs, alphabet) for handwritten answer crops."""
        if self.hand:
            return self.models.hand_log_probs(crops), HAND_ALPHABET
        return self.log_probs(crops), ALPHABET

    def is_ans(self, cell):
        text, conf = self(cell if isinstance(cell, list) else [cell], printed=True)[0]
        return text == "ANS" and conf >= 0.5

    def log_probs(self, crops, printed=False):
        return self.models.reader_log_probs(np.stack([normalize(c, printed) for c in crops])[:, None])

    def __call__(self, crops, printed=False):
        return ctc.decode(self.log_probs(crops, printed), ALPHABET) if crops else []


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


def judge(lp, correct, alphabet):
    """Read one answer from its log-probs and decide right / wrong / review against the correct answer."""
    (ans, conf), = ctc.decode(lp, alphabet)
    given = as_number(ans)
    if not ans:
        return "blank", ans, conf
    if given is None:
        return "review", ans, conf
    read = fmt(given)
    alt = float(np.exp(ctc.string_logprob(lp, [fmt(correct)], alphabet) - ctc.string_logprob(lp, [ans], alphabet))[0])
    return decide(read, conf, fmt(correct), alt), ans, conf


def has_ans(table, xs, reader):
    try:
        col_ys, _, lines = table_rows(table, xs)
    except ValueError:
        return False
    return any(reader.is_ans(cell) for _, cell in label_cells(table, xs, col_ys, lines))


def worksheet_tables(img, reader, report=None):
    """Straightened worksheet tables (table, xs), top to bottom, with the photo the right way up.

    The photo is upright when the label column's 'ANS' can be read; otherwise it is turned 180°.
    Tables without an 'ANS' label (book edge, title box) are dropped."""
    img = fix_orientation(img)
    for cand in [img, np.ascontiguousarray(np.rot90(img, 2))]:
        tables = [g for g in (grid_lines(t) for t in find_tables(cand)) if len(g[1]) >= 4]
        found = [(t, xs) for t, xs in tables if has_ans(t, xs, reader)]
        if found:
            # A full worksheet-like grid whose 'ANS' could not be read is a table we are skipping: say so.
            missed = sum(1 for t, xs in tables if len(xs) >= 10 and not any(t is f for f, _ in found))
            if report is not None and missed:
                report["unread_tables"] = report.get("unread_tables", 0) + missed
            return found
    return []


def short_numbers(qs):
    """Questions with a printed number that has fewer digits than nearly all others in its row - a sign
    that part of it was cut off (e.g. at a curled page edge). Rows are aligned from the bottom."""
    flagged = set()
    depth = max((len(q["numbers"]) for q in qs), default=0)
    for r in range(1, depth + 1):
        row = [(i, q["numbers"][-r][1]) for i, q in enumerate(qs) if len(q["numbers"]) >= r]
        digits = [sum(ch.isdigit() for ch in t) for _, t in row]
        if not digits:
            continue
        common = max(set(digits), key=digits.count)
        if common >= 2 and digits.count(common) >= 0.8 * len(digits):
            flagged |= {i for (i, _), d in zip(row, digits) if d < common}
    return flagged


def check_table(table, xs, reader):
    qs = split_table(table, xs, read=lambda cell: reader([cell], printed=True)[0][0], is_ans=reader.is_ans)
    tight, abc = reader.answer_log_probs([q["answer_crops"]["tight"] for q in qs])
    short = short_numbers(qs)
    results, unclear = [], []
    for i, q in enumerate(qs):
        nums = [t for _, t in q["numbers"]]
        p_conf = min((conf for _, conf in reader([c for c, _ in q["numbers"]], printed=True)), default=0)
        correct = evaluate(nums)
        r = {"q": i + 1, "numbers": nums, "correct": correct, "box": q["answer_box"], "printed_conf": p_conf,
             "col_w": float(np.median(np.diff(xs)))}
        (ans, conf), = ctc.decode(tight[:, i : i + 1], abc)
        r["given"], r["conf"] = ans, conf
        results.append(r)
        if (q["missing_number"] or i in short) and ans:
            # Other columns have a number in a row where this one came out blank: a printed number was lost.
            r["status"], r["why"] = "review", "couldn't read the printed numbers"
            continue
        if correct is None or p_conf < PRINTED_MIN_CONF:
            r["status"], r["why"] = ("blank", "") if not ans else ("review", "couldn't read the printed numbers")
            if correct is None:
                r["why"] = r["why"] or "couldn't read the printed numbers"
            continue
        r["status"], r["given"], r["conf"] = judge(tight[:, i : i + 1], correct, abc)
        if r["status"] == "review":
            unclear.append(i)  # decided below, with the wider answer crops
        else:
            finish(r, q)

    # Only unclear answers need the wider crops (both ANS rows / + the strip above): read just those.
    kinds = ["tight", "rows", "above"] if WIDE_CROPS else ["tight"]
    wide = {"tight": tight[:, unclear]}
    if unclear:
        for k in kinds[1:]:
            wide[k] = reader.answer_log_probs([qs[i]["answer_crops"][k] for i in unclear])[0]
    for j, i in enumerate(unclear):
        r, correct = results[i], results[i]["correct"]
        lps = {k: wide[k][:, j : j + 1] for k in kinds}
        verdicts = {k: judge(lps[k], correct, abc) for k in kinds}
        status, ans, conf = verdicts["tight"]
        p_correct = max(float(np.exp(ctc.string_logprob(lps[k], [fmt(correct)], abc)[0])) for k in kinds)
        if ans and p_correct < ABSENT_MAX:
            status = "wrong"  # whatever exactly the child wrote, it is not the correct answer
        elif len(kinds) > 1:
            # Digits may spill out of the first ANS row: use the crop the reader is most sure about.
            best = max(kinds, key=lambda k: verdicts[k][2])
            status, ans, conf = verdicts[best]
            r["crop"] = best
        r["status"], r["given"], r["conf"] = status, ans, conf
        finish(r, qs[i])
    return results


def finish(r, q):
    if r["status"] == "blank" and normalize(q["answer"]).max() > 0:
        r["status"] = "review"  # the reader saw nothing, but there is ink in the answer cell
    if r["status"] == "review":
        r["why"] = f"handwriting unclear (best guess {r['given']})"


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


def check_photo(path, reader, report=None):
    """path: image file, or an already-loaded BGR image.
    report: optional dict; gets 'unread_tables' = worksheet tables found but not readable."""
    img = path if isinstance(path, np.ndarray) else cv2.imread(str(path))
    if SCAN_LOOK:
        img = scan_look(img)
    all_results, panels = [], []
    for ti, (t, xs) in enumerate(worksheet_tables(img, reader, report)):
        try:
            res = check_table(t, xs, reader)
        except ValueError:  # a worksheet table (it has 'ANS') whose rows/columns could not be lined up
            if report is not None:
                report["unread_tables"] = report.get("unread_tables", 0) + 1
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
        report = {}
        results, sheet = check_photo(path, reader, report)
        for _, msg in retake_reasons(cv2.imread(str(path)), results, report):
            print(f"  RETAKE: {msg}")
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
