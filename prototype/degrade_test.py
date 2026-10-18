"""Fake bad photos: degrade the labelled photos step by step and see where the checker starts to suffer.

For each degradation level: answers checked automatically, mis-marked answers, missed tables, the photo
measurements (quality.py) and how sure the reader is about printed numbers.
usage: python degrade_test.py   (writes degrade_results.json)
"""
import json
from collections import Counter
import cv2
import numpy as np
from check import Reader, check_photo
from e2e import score
from quality import measure, retake_reasons

rng = np.random.default_rng(0)
GATE = False  # True: photos the quality check rejects are asked to retake, and not scored


def dark(img, f):
    return np.clip(img.astype(np.float32) * f, 0, 255).astype(np.uint8)


def low_light(img, f):
    """Dim scene, phone brightens it back up: noise grows as light falls."""
    x = img.astype(np.float32) * f + rng.normal(0, 3, img.shape)
    return np.clip(x / f, 0, 255).astype(np.uint8)


def blur(img, s):
    return cv2.GaussianBlur(img, (0, 0), s)


def shake(img, n):
    k = np.zeros((n, n), np.float32)
    cv2.line(k, (0, n // 3), (n - 1, 2 * n // 3), 1, 1)
    return cv2.filter2D(img, -1, k / k.sum())


def glare(img, a):
    """A lamp's reflection: a soft bright patch over the middle of the page."""
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    blob = np.exp(-(((xx - w * 0.55) / (w * 0.18)) ** 2 + ((yy - h * 0.45) / (h * 0.22)) ** 2))[..., None]
    return np.clip(img + (255 - img.astype(np.float32)) * blob * a * 1.6, 0, 255).astype(np.uint8)


def shadow(img, s):
    """Hand / phone shadow: the left part of the page darker, soft edge."""
    h, w = img.shape[:2]
    ramp = np.clip((np.arange(w) - w * 0.4) / (w * 0.08), 0, 1)
    m = (s + (1 - s) * ramp)[None, :, None]
    return np.clip(img * m, 0, 255).astype(np.uint8)


def small(img, f):
    """Photo taken from too far away: fewer pixels on the worksheet."""
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)


def jpeg(img, q):
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


TESTS = {
    "original": (lambda img, _: img, [None]),
    "dark": (dark, [0.6, 0.4, 0.25, 0.15]),
    "low light (noisy)": (low_light, [0.5, 0.25, 0.15, 0.1]),
    "blur": (blur, [1.0, 1.5, 2.0, 3.0]),
    "shake": (shake, [5, 9, 13, 17]),
    "glare": (glare, [0.3, 0.5, 0.7, 0.9]),
    "shadow": (shadow, [0.7, 0.5, 0.35, 0.2]),
    "too far": (small, [0.75, 0.5, 0.35, 0.25]),
    "jpeg": (jpeg, [30, 15, 8, 4]),
}

if __name__ == "__main__":
    import sys
    GATE = "--gate" in sys.argv
    truth = {k: v for k, v in json.load(open("testset.json")).items() if not k.startswith("_")}
    if "--test-only" in sys.argv:  # held-out photos only (the others feed augmentation / training)
        from realset import TEST_PHOTOS
        truth = {k: v for k, v in truth.items() if k in TEST_PHOTOS}
    photos = {f: cv2.imread(f"../{f}") for f in truth}
    old = "--old" in sys.argv
    reader = Reader(weights="reader_v2.pt", hand_head="hand_head.pt") if old else Reader()
    rows = []
    for name, (fn, levels) in TESTS.items():
        for lvl in levels:
            c, metrics, pconf, bad_photos, retake = Counter(), [], [], [], 0
            for f, tables in truth.items():
                img = fn(photos[f], lvl)
                report = {}
                results, _ = check_photo(img, reader, report)
                if GATE and retake_reasons(img, results, report):
                    retake += 1
                    continue
                m = measure(img)
                m["col_w"] = float(np.median([r["col_w"] for r in results])) * 1600 / img.shape[1] if results else 0.0
                p = [r["printed_conf"] for r in results]
                m["printed_unsure"] = float(np.mean(np.array(p) < 0.99)) if p else 1.0
                metrics.append(m)
                pconf += p
                before = c[("child right", "wrong")] + c[("child wrong", "right")]
                score(f, tables, results, c, verbose=False)
                if c[("child right", "wrong")] + c[("child wrong", "right")] > before:
                    bad_photos.append({"photo": f, **m})
            n = max(sum(v for k, v in c.items() if isinstance(k, tuple)) + c["missed"], 1)
            review = sum(v for k, v in c.items() if isinstance(k, tuple) and k[1] == "review")
            auto = sum(v for k, v in c.items() if isinstance(k, tuple) and k[1] in ("right", "wrong"))
            bad = c[("child right", "wrong")] + c[("child wrong", "right")]
            if not metrics:
                metrics = [measure(photos[next(iter(truth))])]
            row = {"test": name, "level": lvl, "auto": auto / n, "review": review / n, "missed": c["missed"] / n,
                   "mis_marked": bad, "printed_unsure": float(np.mean(np.array(pconf) < 0.99)) if pconf else 1.0,
                   "bad_photos": bad_photos, "retake": retake / len(truth),
                   **{k: float(np.median([m[k] for m in metrics])) for k in metrics[0]}}
            rows.append(row)
            print(f"{name:18} {str(lvl):5} retake {row['retake']:4.0%} | accepted photos: auto {row['auto']:4.0%}  review {row['review']:4.0%}  "
                  f"tables missed {row['missed']:4.0%}  mis-marked {bad}  printed unsure {row['printed_unsure']:4.0%} | "
                  + "  ".join(f"{k} {row[k]:.3g}" for k in metrics[0]), flush=True)
    tag = "_".join(a.strip("-") for a in sys.argv[1:]) or "plain"
    json.dump(rows, open(f"degrade_results_{tag}.json", "w"), indent=1)
