"""Shared preprocessing: any cell crop (real or synthetic) -> fixed-size ink map the model sees."""
import cv2
import numpy as np

H, W = 32, 128
ALPHABET = "0123456789-.×÷+IVX"  # IVX = printed column headers. CTC blank is index 0, so char i -> i + 1


def remove_lines(ink):
    """Zero out leftover grid-line fragments: straight strokes spanning most of the crop."""
    h, w = ink.shape
    _, bw = cv2.threshold(np.clip(ink, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.4), 3), 1)))
    vl = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * 0.85), 3))))
    lines = cv2.dilate(hl | vl, np.ones((5, 5), np.uint8))
    ink[lines > 0] = 0
    return ink


def to_ink(gray):
    """Grey crop -> float ink map in [0, 1] (0 = paper, 1 = darkest ink), robust to lighting."""
    g = gray.astype(np.float32)
    bg = cv2.medianBlur(gray, 21).astype(np.float32) if min(gray.shape) > 21 else np.full_like(g, np.percentile(g, 80))
    ink = remove_lines(np.clip(bg - g, 0, None))
    hi = np.percentile(ink, 99.5)
    if hi < 20:  # nothing darker than the paper: empty cell
        return np.zeros_like(g)
    return np.clip(ink / hi, 0, 1)


def main_cluster(stats, keep):
    """Components forming one printed number: start from the biggest, add neighbours whose horizontal gap
    is under half a glyph height. Marks standing apart (a child's pen stroke beside the number) are left out."""
    big = max(keep, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    gh = stats[big, cv2.CC_STAT_HEIGHT]
    x0, x1 = stats[big, 0], stats[big, 0] + stats[big, 2]
    cluster, rest = [big], [i for i in keep if i != big]
    grew = True
    while grew:
        grew = False
        for i in list(rest):
            l, r = stats[i, 0], stats[i, 0] + stats[i, 2]
            if max(l - x1, x0 - r) < 0.5 * gh:
                cluster.append(i); rest.remove(i); x0, x1 = min(x0, l), max(x1, r); grew = True
    return cluster


def normalize(gray, printed=False):
    """Grey crop -> (H, W) float32 model input, glyphs tightly cropped and centred.
    printed=True: keep only the main printed number, ignoring marks that stand apart from it."""
    ink = to_ink(gray)
    mask = (ink > 0.35).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    ch, cw = mask.shape

    def is_glyph(i):
        x, y, w, h, area = stats[i]
        if area < max(4, mask.size * 0.0015):
            return False  # speck
        if w > cw * 0.45 and h < w * 0.2:
            return False  # flat stroke wider than a minus sign: leftover (curled) grid line or underline
        if h > ch * 0.9 and w < h * 0.15 and (x <= 1 or x + w >= cw - 1):
            return False  # vertical grid-line fragment at the crop edge
        return True

    keep = [i for i in range(1, n) if is_glyph(i)]
    if keep:  # small marks cut off at the top edge are the bottom of the printed number in the row above
        big = max(stats[i, cv2.CC_STAT_AREA] for i in keep)
        keep = [i for i in keep if not (stats[i, cv2.CC_STAT_TOP] <= 1 and stats[i, cv2.CC_STAT_AREA] < 0.25 * big)]
    if keep and printed:
        keep = main_cluster(stats, keep)
    out = np.zeros((H, W), np.float32)
    if not keep:
        return out
    x0 = min(stats[i, 0] for i in keep); y0 = min(stats[i, 1] for i in keep)
    x1 = max(stats[i, 0] + stats[i, 2] for i in keep); y1 = max(stats[i, 1] + stats[i, 3] for i in keep)
    m = 2
    crop = ink[max(y0 - m, 0) : y1 + m, max(x0 - m, 0) : x1 + m]
    ch, cw = crop.shape
    s = min((H - 4) / ch, (W - 4) / cw)
    r = cv2.resize(crop, (max(int(cw * s), 1), max(int(ch * s), 1)), interpolation=cv2.INTER_AREA)
    oy, ox = (H - r.shape[0]) // 2, (W - r.shape[1]) // 2
    out[oy : oy + r.shape[0], ox : ox + r.shape[1]] = r
    return out
