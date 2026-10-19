"""Real labelled cells from the photos (testset.json): used to fine-tune and to measure accuracy honestly.

Photos are split into train / test by photo, so test photos are never seen in training.
"""
import json
import cv2
import numpy as np
from cells import split_table
from norm import normalize

TEST_PHOTOS = [  # held out: never used for training (re-shots of a held-out page are held out too)
    "WhatsApp Image 2026-10-16 at 15.32.22.jpeg",
    "WhatsApp Image 2026-10-16 at 22.37.55 (2).jpeg",
    "WhatsApp Image 2026-10-16 at 22.37.56 (2).jpeg",
    "WhatsApp Image 2026-10-16 at 22.37.56 (3).jpeg",
    "19022026/WhatsApp Image 2026-10-19 at 14.42.06.jpeg",      # re-shot of 22.37.56 (3)
    "19022026/WhatsApp Image 2026-10-19 at 14.42.07 (3).jpeg",  # re-shot of 22.37.55 (2)
    "19022026/WhatsApp Image 2026-10-19 at 14.42.09 (1).jpeg",  # re-shot of 15.32.22
    "19022026/WhatsApp Image 2026-10-19 at 14.42.06 (2).jpeg",  # new page, 2-digit numbers
    "19022026/WhatsApp Image 2026-10-19 at 14.42.08 (2).jpeg",
    "19022026/WhatsApp Image 2026-10-19 at 14.42.09 (2).jpeg",
]

CALIB_PHOTOS = [  # kept out of training too: used only to calibrate confidence (never the test photos)
    "WhatsApp Image 2026-10-16 at 22.37.56 (1).jpeg",
    "19022026/WhatsApp Image 2026-10-19 at 14.42.07 (1).jpeg",
    "19022026/WhatsApp Image 2026-10-19 at 14.42.09 (3).jpeg",
]


def split_of(photo):
    return "test" if photo in TEST_PHOTOS else "calib" if photo in CALIB_PHOTOS else "train"


def has_ink(cell):
    return "1" if normalize(cell).max() > 0 else ""


def cells_from_image(img, fname, tables, reader, verify=False):
    """Labelled cells of one photo: list of (normalized_cell, label, kind, photo, tag, raw_crop).

    verify=True (augmented photos): drop any table whose layout does not match the transcription, or whose
    printed numbers mostly disagree with it - a sign the checker misread the grid, so labels would be wrong.
    """
    from check import worksheet_tables  # imported here: check.py is the heavier module
    found = [g for g in worksheet_tables(img, reader) if len(g[1]) >= 11]
    if len(found) != len(tables):
        return []
    out = []
    for ti, ((t, xs), gt) in enumerate(zip(found, tables)):
        try:
            qs = split_table(t, xs, read=has_ink, is_ans=reader.is_ans)  # every inked row; align from the bottom
        except ValueError:
            continue
        rows = []
        for qi, (q, nums, ans) in enumerate(zip(qs, gt["numbers"], gt["answers"])):
            tag = f"{fname[-14:-5]} T{ti + 1} Q{qi + 1}"
            cells = q["numbers"][-len(nums):]
            if len(cells) < len(nums):
                rows = None
                break
            for ri, ((cell, _), n) in enumerate(zip(cells, nums)):
                if n != "?":
                    rows.append((normalize(cell), str(n), "printed", fname, f"{tag} r{ri + 1}", cell))
            if ans not in ("?", ""):  # "" = left unanswered
                rows.append((normalize(q["answer"]), ans, "hand", fname, tag, q["answer"]))
        if rows is None or (verify and len(qs) != len(gt["numbers"])):
            continue
        if verify:
            printed = [r for r in rows if r[2] == "printed"]
            reads = reader([r[5] for r in printed], printed=True)
            if np.mean([p == r[1] for (p, _), r in zip(reads, printed)]) < 0.8:
                continue
        out += rows
    return out


def load(path="testset.json", root=".."):
    """Labelled cells of every transcribed photo."""
    from check import Reader
    reader = Reader()
    out = []
    for fname, tables in json.load(open(path)).items():
        if fname.startswith("_"):
            continue
        cells = cells_from_image(cv2.imread(f"{root}/{fname}"), fname, tables, reader)
        if not cells:
            print(f"  ! {fname}: table layout does not match the transcription - skipped")
        out += cells
    return out


if __name__ == "__main__":
    data = load()
    test = np.array([d[3] in TEST_PHOTOS for d in data])
    kind = np.array([d[2] for d in data])
    for split, m in [("train", ~test), ("test", test)]:
        print(f"{split}: {((kind == 'printed') & m).sum()} printed, {((kind == 'hand') & m).sum()} handwritten")
    np.savez_compressed("realset.npz", x=np.stack([d[0] for d in data]), y=[d[1] for d in data], kind=kind,
                        test=test, tag=[d[4] for d in data])
