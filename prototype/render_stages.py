"""Save a picture of every stage of the checker, for one photo and a few chosen answers.

usage: python render_stages.py <photo> <out_dir> [table:column ...]   e.g.  3:2 3:4   (1-based)
"""
import sys
from pathlib import Path
import cv2
import numpy as np
import ctc
from cells import split_table, table_rows
from check import Reader, check_photo, worksheet_tables
from grid import angled_open, binarize, column_curves, find_tables, fix_orientation
from norm import normalize

photo, out = sys.argv[1], Path(sys.argv[2])
picks = [tuple(int(v) for v in p.split(":")) for p in sys.argv[3:]] or [(1, 1)]
out.mkdir(parents=True, exist_ok=True)
reader = Reader()


def save(name, img, width=None):
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if width and img.shape[1] != width:
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(out / name), img)


def label(img, text):
    if img.dtype != np.uint8:  # model input: ink = 1 on 0 background -> dark ink on white
        img = ((1 - np.clip(img, 0, 1)) * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    img = cv2.copyMakeBorder(img, 26, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(img, text, (4, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1, cv2.LINE_AA)
    return img


def row(imgs, h=130):
    imgs = [cv2.resize(i, (int(i.shape[1] * h / i.shape[0]), h), interpolation=cv2.INTER_NEAREST) for i in imgs]
    imgs = [cv2.cvtColor(i, cv2.COLOR_GRAY2BGR) if i.ndim == 2 else i for i in imgs]
    return np.hstack([cv2.copyMakeBorder(i, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)) for i in imgs])


# 1. the photo as it arrives
img = cv2.imread(photo)
save("01_photo.jpg", img)

# 2. turned so table lines run sideways (upside-down is settled later by finding 'ANS')
up = fix_orientation(img)
save("02_orientation.jpg", up)

# 3. long straight strokes = candidate grid lines (only lines crossing other lines are kept)
gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
H, W = gray.shape
bw = binarize(gray)
hz = angled_open(bw, max(W // 30, 20), range(-15, 16, 3), vertical=False)
vt = angled_open(bw, max(H // 30, 15), range(-20, 21, 4), vertical=True)
vis = up.copy()
vis[hz > 0] = (255, 0, 0)
vis[vt > 0] = (0, 0, 255)
save("03_line_pixels.jpg", np.hstack([cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR), vis]))

# 4. tables found (groups of crossing lines), each un-skewed from its 4 corners
tables = find_tables(up)
save("04_tables_found.jpg", np.vstack([cv2.copyMakeBorder(t, 6, 6, 0, max(x.shape[1] for x in tables) - t.shape[1],
                                                          cv2.BORDER_CONSTANT, value=(255, 255, 255)) for t in tables]))

# 5. column lines traced as curves (page curl), then every column straightened
found = worksheet_tables(img, reader)
for ti, t in enumerate(tables):
    v = t.copy()
    y = np.arange(t.shape[0])
    for f in column_curves(t):
        cv2.polylines(v, [np.stack([f(y), y], 1).astype(np.int32)], False, (0, 0, 255), 2)
    save(f"05_t{ti + 1}_a_column_curves.jpg", v)
for ti, (t, xs) in enumerate(found):
    v = t.copy()
    col_ys, pitch, lines = table_rows(t, xs)
    for x in xs:
        cv2.line(v, (x, 0), (x, t.shape[0]), (0, 0, 255), 2)
    for i, ys in enumerate(col_ys):
        for yy in ys:
            cv2.line(v, (xs[i], yy), (xs[i + 1], yy), (255, 0, 0), 2)
    save(f"05_t{ti + 1}_b_straightened_rows_columns.jpg", v)
    # 6. the mask of grid-line pixels that gets painted out before reading
    m = t.copy()
    m[lines > 0] = (0, 200, 255)
    save(f"06_t{ti + 1}_line_mask_to_erase.jpg", m)

# 7-9. for each chosen answer: cells, cleaning, model input, readings, decision
results, sheet = check_photo(img, reader)
for ti, ci in picks:
    t, xs = found[ti - 1]
    qs = split_table(t, xs, read=lambda c: reader([c], printed=True)[0][0], is_ans=reader.is_ans)
    q = qs[ci - 1]
    x0, x1 = q["answer_box"][0], q["answer_box"][2]
    tag = f"T{ti}_col{ci}"
    # 7. the column in the straightened table: raw pixels vs after grid lines are painted out
    col_ys, pitch, lines = table_rows(t, xs)
    raw = t[:, x0:x1]
    gray_col = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    inpainted = cv2.inpaint(gray_col, cv2.dilate(lines[:, x0:x1], np.ones((3, 3), np.uint8)), 3, cv2.INPAINT_TELEA)
    save(f"07_{tag}_column_raw_vs_lines_erased.png", row([label(gray_col, "raw"), label(inpainted, "lines erased")], 420))
    # 8. printed cells: cleaned crop -> what the printed-number model sees -> reading
    tiles = []
    for cell, txt in q["numbers"]:
        (rd, cf), = reader([cell], printed=True)
        tiles += [label(cell, "cell"), label(normalize(cell, True), f"model input -> '{rd}' {cf:.2f}")]
    save(f"08_{tag}_printed_cells.png", row(tiles, 110))
    # 9. the three answer views: crop (lines erased) -> what each model sees -> reading
    tiles = []
    for k in ["tight", "rows", "above"]:
        crop = q["answer_crops"][k]
        lp, abc = reader.answer_log_probs([crop])
        (rd, cf), = ctc.decode(lp, abc)
        tiles += [label(crop, f"{k}: '{rd}' {cf:.2f}")]
    save(f"09_{tag}_answer_views.png", row(tiles, 170))
    r = next(x for x in results if x["table"] == ti and x["q"] == ci)
    print(f"{tag}: printed {r['numbers']} -> correct {r['correct']}; child read '{r['given']}' -> {r['status']}"
          + (f" ({r['why']})" if r.get("why") else ""))

# 10. the marked sheet
save("10_marked_sheet.jpg", sheet)
print("saved to", out)
