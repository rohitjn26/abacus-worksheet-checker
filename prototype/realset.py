"""Real labelled cells from the photos (testset.json): used to fine-tune and to measure accuracy honestly.

Photos are split into train / test by photo, so test photos are never seen in training.
"""
import json
import cv2
import numpy as np
from cells import split_table
from norm import normalize

TEST_PHOTOS = [  # held out: never used for training
    "WhatsApp Image 2026-10-16 at 15.32.22.jpeg",
    "WhatsApp Image 2026-10-16 at 22.37.55 (2).jpeg",
    "WhatsApp Image 2026-10-16 at 22.37.56 (2).jpeg",
    "WhatsApp Image 2026-10-16 at 22.37.56 (3).jpeg",
]


def has_ink(cell):
    return "1" if normalize(cell).max() > 0 else ""


def load(path="testset.json", root=".."):
    """Returns list of (normalized_cell, label, kind, photo, tag, raw_crop); kind is 'printed' or 'hand'."""
    from check import Reader, worksheet_tables  # imported here: check.py is the heavier module
    reader = Reader()
    truth = json.load(open(path))
    out = []
    for fname, tables in truth.items():
        if fname.startswith("_"):
            continue
        found = [g for g in worksheet_tables(cv2.imread(f"{root}/{fname}"), reader) if len(g[1]) >= 11]
        if len(found) != len(tables):
            print(f"  ! {fname}: found {len(found)} tables, truth has {len(tables)} - skipped")
            continue
        for ti, ((t, xs), gt) in enumerate(zip(found, tables)):
            qs = split_table(t, xs, read=has_ink, is_ans=reader.is_ans)  # keep every inked row; align to truth from the bottom
            for qi, (q, nums, ans) in enumerate(zip(qs, gt["numbers"], gt["answers"])):
                tag = f"{fname[-14:-5]} T{ti + 1} Q{qi + 1}"
                cells = q["numbers"][-len(nums):]
                for ri, ((cell, _), n) in enumerate(zip(cells, nums)):
                    if n != "?":
                        out.append((normalize(cell), str(n), "printed", fname, f"{tag} r{ri + 1}", cell))
                if ans != "?":
                    out.append((normalize(q["answer"]), ans, "hand", fname, tag, q["answer"]))
    return out


if __name__ == "__main__":
    data = load()
    test = np.array([d[3] in TEST_PHOTOS for d in data])
    kind = np.array([d[2] for d in data])
    for split, m in [("train", ~test), ("test", test)]:
        print(f"{split}: {((kind == 'printed') & m).sum()} printed, {((kind == 'hand') & m).sum()} handwritten")
    np.savez_compressed("realset.npz", x=np.stack([d[0] for d in data]), y=[d[1] for d in data], kind=kind,
                        test=test, tag=[d[4] for d in data])
