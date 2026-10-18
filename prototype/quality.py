"""Photo-quality measurements, used to tell a parent to retake a photo.

All measured on the grey photo resized to a fixed width, so numbers are comparable across phones.
"""
import cv2
import numpy as np

REF_W = 1200


def measure(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (REF_W, int(gray.shape[0] * REF_W / gray.shape[1])), interpolation=cv2.INTER_AREA)
    g = gray.astype(np.float32)
    # Paper brightness and its unevenness: a smooth "background" with the ink removed (closing = max filter).
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((15, 15), np.uint8)), 51).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    return {
        "brightness": float(np.percentile(bg, 50)),                          # typical paper brightness, 0-255
        "contrast": float(np.percentile(g, 95) - np.percentile(g, 2)),       # paper vs darkest ink
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),           # edge strength
        "glare": float((gray >= 250).mean()),                                # share of blown-out pixels
        "shadow": float(np.percentile(bg, 5) / max(np.percentile(bg, 95), 1)),  # darkest/brightest paper (1 = even)
        "noise": float(np.median(np.abs(hp - np.median(hp))) * 1.4826),     # robust noise level
    }


# Retake rules. Thresholds from degrade_test.py: each sits where wrong marks or failures start, with a
# margin from the worst of the 10 good sample photos.
RULES = [
    ("no_table", "Couldn't find the worksheet. Make sure the whole table is in the photo."),
    ("partial", "Part of the sheet couldn't be read. Keep the page flat and the whole sheet in the photo."),
    ("dark", "Photo is too dark. Move to a brighter place."),
    ("grainy", "Photo is grainy (too little light). Move to a brighter place."),
    ("blurry", "Photo is blurry. Hold the phone steady and tap to focus."),
    ("glare", "Light is reflecting off the page. Tilt the phone or move away from the lamp."),
    ("shadow", "There is a strong shadow on the page. Move so your shadow is not on the sheet."),
    ("far", "Too far away. Move the phone closer so the table fills the photo."),
    ("unclear", "Photo is not clear enough to read the numbers. Please retake it."),
]


def retake_reasons(img, results, report=None):
    """Reasons a parent should retake the photo ([] = photo is fine).
    results, report: output of check.check_photo(img, reader, report)."""
    m = measure(img)
    found = []
    if not results:
        found.append("no_table")
    elif report and report.get("unread_tables"):
        found.append("partial")
    if m["brightness"] < 60:
        found.append("dark")
    if m["noise"] > 5:
        found.append("grainy")
    if m["sharpness"] < 50:
        found.append("blurry")
    if m["glare"] > 0.005:
        found.append("glare")
    if m["shadow"] < 0.25:
        found.append("shadow")
    if results:
        if np.median([r["col_w"] for r in results]) < 95:  # pixels per column in this photo (good photos: 111+)
            found.append("far")
        if np.mean([r["printed_conf"] < 0.99 for r in results]) > 0.25:  # good photos: up to 0.10
            found.append("unclear")
    return [(k, msg) for k, msg in RULES if k in found]
