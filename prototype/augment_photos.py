"""Whole-photo augmentation: random 'bad photo' versions of the TRAINING photos (never the held-out ones),
run through the checker, cells labelled from the transcription. Bad variants whose grid is misread are
dropped (see realset.cells_from_image verify=True).

usage: python augment_photos.py <variants per photo> <raw_out_dir>
  -> augset.npz (normalized cells for our model) and <raw_out_dir>/aug/*.png + aug.txt (raw crops, PaddleOCR)
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from check import Reader
from degrade_test import blur, dark, glare, jpeg, low_light, shadow, shake, small
from realset import CALIB_PHOTOS, TEST_PHOTOS, cells_from_image

rng = np.random.default_rng(1)


def tilt(img):
    """Phone not held flat: random perspective (corners moved up to 6%) and rotation (up to 4 degrees)."""
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + rng.uniform(-0.06, 0.06, (4, 2)) * [w, h]).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    R = np.vstack([cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-4, 4), 1.0), [0, 0, 1]])
    return cv2.warpPerspective(img, R @ M, (w, h), borderMode=cv2.BORDER_REPLICATE)


# Ranges a photo-quality check would still accept (worse photos get a retake request instead).
EFFECTS = [
    lambda im: dark(im, rng.uniform(0.4, 1.15)),
    lambda im: low_light(im, rng.uniform(0.2, 0.6)),
    lambda im: blur(im, rng.uniform(0.5, 1.2)),
    lambda im: shake(im, int(rng.choice([3, 5]))),
    lambda im: glare(im, rng.uniform(0.1, 0.45)),
    lambda im: shadow(im, rng.uniform(0.4, 0.85)),
    lambda im: small(im, rng.uniform(0.75, 1.0)),
    lambda im: jpeg(im, int(rng.integers(20, 90))),
    tilt,
]


def bad_photo(img):
    for i in rng.choice(len(EFFECTS), size=int(rng.integers(1, 4)), replace=False):
        img = EFFECTS[i](img)
    return img


if __name__ == "__main__":
    n_var, raw_dir = int(sys.argv[1]), Path(sys.argv[2])
    (raw_dir / "aug").mkdir(parents=True, exist_ok=True)
    truth = {k: v for k, v in json.load(open("testset.json")).items()
             if not k.startswith("_") and k not in TEST_PHOTOS and k not in CALIB_PHOTOS}
    reader = Reader()
    data, kept, lines = [], 0, []
    for f, tables in truth.items():
        img = cv2.imread(f"../{f}")
        for v in range(n_var):
            cells = cells_from_image(bad_photo(img), f, tables, reader, verify=True)
            kept += bool(cells)
            for x, label, kind, _, tag, raw in cells:
                name = f"aug/{len(data):06d}.png"
                cv2.imwrite(str(raw_dir / name), raw)
                lines.append(f"{name}\t{label}\t{kind}\t{f}")
                data.append((x, label, kind))
        print(f"{f[-14:-5]}: done", flush=True)
    (raw_dir / "aug.txt").write_text("\n".join(lines) + "\n")
    kinds = np.array([d[2] for d in data])
    np.savez_compressed("augset.npz", x=np.stack([d[0] for d in data]), y=[d[1] for d in data], kind=kinds)
    print(f"variants with usable tables: {kept}/{n_var * len(truth)};  cells: {(kinds == 'printed').sum()} printed, "
          f"{(kinds == 'hand').sum()} handwritten")
