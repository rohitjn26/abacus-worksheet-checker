"""Find worksheet tables in a photo, straighten them, and locate their grid lines."""
import cv2
import numpy as np

LOW_CONTRAST = 120  # photos with less contrast than this (p98 - p2 grey levels) are stretched before finding lines


def fix_orientation(img):
    """Turn the photo so table lines run mainly horizontally (tables are wider than tall).
    Upright vs upside down is settled later by finding the 'ANS' label (see check.worksheet_tables)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    horiz, vert = line_masks(gray)
    return np.ascontiguousarray(np.rot90(img)) if vert.sum() > horiz.sum() else img


def line_masks(gray, hlen=None, vlen=None):
    """Binary masks of horizontal / vertical strokes at least hlen / vlen pixels long."""
    bw = binarize(gray)
    h, w = gray.shape
    hlen, vlen = hlen or max(w // 30, 20), vlen or max(h // 30, 15)
    horiz = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (int(hlen), 1)))
    vert = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, int(vlen))))
    return horiz, vert


def order_corners(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def crossing_lines(horiz, vert, min_crossings=3):
    """Keep only line pieces that cross at least `min_crossings` lines of the other direction.
    Table lines cross many; a page edge, the book's fold or a stray long stroke cross few."""
    k = np.ones((3, 3), np.uint8)
    n_x, x_lab = cv2.connectedComponents(cv2.dilate(horiz, k) & cv2.dilate(vert, k))
    out = []
    for mask in (horiz, vert):
        n, lab = cv2.connectedComponents(cv2.dilate(mask, k))
        both = (lab > 0) & (x_lab > 0)
        pairs = np.unique(lab[both].astype(np.int64) * (n_x + 1) + x_lab[both])
        counts = np.bincount(pairs // (n_x + 1), minlength=n)
        out.append(np.where(counts[lab] >= min_crossings, mask, 0).astype(np.uint8))
    return out[0], out[1]


def find_tables(img):
    """Return straightened crops of each table, top to bottom.

    A table is the group of grid-line blobs that sit side by side (a faint outline can split one table
    into pieces). Its 4 corners are the extreme points of the group, so skewed photos un-skew properly.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    bw = binarize(gray)
    horiz = angled_open(bw, max(W // 30, 20), range(-15, 16, 3), vertical=False)
    vert = angled_open(bw, max(H // 30, 15), range(-20, 21, 4), vertical=True)
    horiz, vert = crossing_lines(horiz, vert)
    grid = cv2.dilate(horiz | vert, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = [(c, cv2.boundingRect(c)) for c in contours if cv2.boundingRect(c)[3] > H * 0.05]
    big = [(c, r) for c, r in rects if r[2] > W * 0.08]
    thin = [(c, r) for c, r in rects if r[2] <= W * 0.08]

    groups = []
    for c, (x, y, w, h) in sorted(big, key=lambda cr: cr[1][1]):
        for g in groups:
            gy0, gy1 = g["y"]
            x_gap = max(x - g["x"][1], g["x"][0] - (x + w))  # < 0 when they overlap horizontally
            # Same rows and touching / nearly touching: pieces of one table (a faint outline split it).
            # Not joined across a real gap, e.g. the facing page's table across the book's fold.
            if min(gy1, y + h) - max(gy0, y) > 0.5 * min(gy1 - gy0, h) and x_gap < W * 0.02:
                g["pts"].append(c); g["y"] = (min(gy0, y), max(gy1, y + h)); g["x"] = (min(g["x"][0], x), max(g["x"][1], x + w))
                break
        else:
            groups.append({"pts": [c], "y": (y, y + h), "x": (x, x + w)})
    # Thin pieces (e.g. an outer border line cut off from the grid) only join a table they sit beside.
    for c, (x, y, w, h) in thin:
        for g in groups:
            gy0, gy1 = g["y"]
            inside = min(gy1, y + h) - max(gy0, y) > 0.8 * h and h > 0.5 * (gy1 - gy0)
            near = g["x"][0] - 40 < x < g["x"][1] + 40
            if inside and near:
                g["pts"].append(c)
                break

    tables = []
    for g in groups:
        pts = np.vstack(g["pts"]).reshape(-1, 2).astype(np.float32)
        if np.ptp(pts[:, 0]) < W * 0.3 or np.ptp(pts[:, 1]) < H * 0.08:
            continue
        s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
        src = np.array([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]], np.float32)
        tw = int(max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2])))
        th = int(max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2])))
        dst = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        tables.append((g["y"][0], cv2.warpPerspective(img, M, (tw, th), borderValue=(255, 255, 255))))
    return [t for _, t in sorted(tables, key=lambda t: t[0])]


def line_positions(profile, min_frac, length):
    """Centres of runs in a projection profile that exceed min_frac * length."""
    on = profile > min_frac * length * 255
    pos, start = [], None
    for i, v in enumerate(np.append(on, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            pos.append((start + i - 1) // 2)
            start = None
    return pos


def angled_open(bw, length, angles, vertical):
    """Keep straight strokes at least `length` long at any of the given angles (degrees from axis)."""
    out = np.zeros_like(bw)
    for a in angles:
        k = np.zeros((length, length), np.uint8)
        c, r, t = length // 2, length / 2, np.deg2rad(a)
        dx, dy = (np.sin(t) * r, np.cos(t) * r) if vertical else (np.cos(t) * r, np.sin(t) * r)
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
        out |= cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)
    return out


def binarize(gray):
    """Ink / line pixels; only used for finding lines, never fed to a model. Good photos are binarized as
    they are; a low-contrast (dark / shadowed) photo is first stretched to normal contrast."""
    g = gray.astype(np.float32)
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    if hi - lo < LOW_CONTRAST:
        g = np.clip((g - lo) * (LOW_CONTRAST / max(hi - lo, 1)) + lo, 0, 255)
    return cv2.adaptiveThreshold(g.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 15)


def column_curves(table):
    """Column lines as curves x = f(y) (quadratic), left to right. Tolerates lean and page curl."""
    gray = cv2.cvtColor(table, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    vert = angled_open(binarize(gray), max(h // 8, 15), range(-20, 21, 5), vertical=True)
    vert = cv2.dilate(vert, np.ones((9, 3), np.uint8))  # bridge small breaks
    n, labels, stats, _ = cv2.connectedComponentsWithStats(vert)
    curves = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_HEIGHT] < 0.4 * h:
            continue
        ys, xs = np.nonzero(labels == i)
        f = np.poly1d(np.polyfit(ys, xs, 2 if np.ptp(ys) > 0.6 * h else 1))
        curves.append(f)
    curves.sort(key=lambda f: f(h / 2))
    merged = []
    for f in curves:
        if merged and f(h / 2) - merged[-1](h / 2) < 15:
            continue
        merged.append(f)
    return merged


def rectify_columns(table, curves):
    """Resample so every column line becomes perfectly vertical. Returns (image, xs)."""
    h = table.shape[0]
    y = np.arange(h, dtype=np.float32)
    strips, xs = [], [0]
    for f, g in zip(curves, curves[1:]):
        left, right = f(y), g(y)
        cw = int(round(np.median(right - left)))
        if cw < 5:
            continue
        u = np.arange(cw, dtype=np.float32) / cw
        map_x = (left[:, None] + u[None, :] * (right - left)[:, None]).astype(np.float32)
        map_y = np.repeat(y[:, None], cw, axis=1)
        strips.append(cv2.remap(table, map_x, map_y, cv2.INTER_LINEAR, borderValue=(255, 255, 255)))
        xs.append(xs[-1] + cw)
    return np.hstack(strips), xs


def grid_lines(table):
    """Straighten a table column by column. Returns (rectified table, column-line xs)."""
    curves = column_curves(table)
    if len(curves) < 3:
        return table, []
    # An outer border too faint/cut to detect: if a full column's width of table lies beyond the last
    # (or before the first) line, add the border one median column width away.
    h, w = table.shape[:2]
    mids = [f(h / 2) for f in curves]
    m = float(np.median(np.diff(mids)))
    # The crop's own edge is the table's outer edge (find_tables uses its extreme points), so when about
    # one column's width remains, the border is at the crop edge - not at a guessed median width, which
    # can cut off digits where the page curls.
    if w - mids[-1] > 0.6 * m:
        rest = w - 1 - mids[-1]
        curves.append(curves[-1] + np.poly1d([rest if rest < 1.5 * m else m]))
    if mids[0] > 0.6 * m:
        curves.insert(0, curves[0] - np.poly1d([mids[0] if mids[0] < 1.5 * m else m]))
    # Columns are equal width: a gap of ~k columns means k-1 lines were missed (faint, dark or blurred);
    # put them back evenly. Otherwise two columns merge and one child's answer is read with another's.
    filled = curves[:1]
    for f in curves[1:]:
        gap = f(h / 2) - filled[-1](h / 2)
        k = round(gap / m)
        if k >= 2 and abs(gap / k - m) < 0.25 * m:
            filled += [filled[-1] + (f - filled[-1]) * (j / k) for j in range(1, k)]
        filled.append(f)
    return rectify_columns(table, filled)
