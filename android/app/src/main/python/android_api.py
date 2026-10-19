"""Entry point the Android app calls (through Chaquopy). The checker itself is the tested code from
prototype/; only the model backend differs: models run in Kotlin (ONNX Runtime for Android).
"""
import json
import cv2
import numpy as np
from java import jclass
from check import Reader, check_photo
from quality import retake_reasons

MAX_SIDE = 1600  # the checker's thresholds were set on ~1600-pixel photos; phone cameras give 4000+


class JavaBackend:
    """runtime.Models backend: hands each model call to com.abacus.checker.OnnxBridge (Kotlin)."""

    def __init__(self):
        self.bridge = jclass("com.abacus.checker.OnnxBridge")

    def run(self, model_file, input_name, x):
        x = np.ascontiguousarray(x, np.float32)
        res = self.bridge.run(model_file, input_name, x.tobytes(), [int(d) for d in x.shape])
        data = np.frombuffer(bytes(res.getData()), np.float32)
        return data.reshape([int(d) for d in res.getShape()])


_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        _reader = Reader(backend=JavaBackend())
    return _reader


def warm_up():
    """Load Python modules and models ahead of the first photo."""
    reader = _get_reader()
    blank = np.full((40, 120), 230, np.uint8)
    reader([blank], printed=True)
    reader.answer_log_probs([blank])


def check(path, out_path):
    """Check one photo. Writes the marked sheet to out_path; returns JSON with counts and retake reasons."""
    img = cv2.imread(path)
    if img is None:
        return json.dumps({"error": "Could not open the photo."})
    s = MAX_SIDE / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    report = {}
    results, sheet = check_photo(img, _get_reader(), report)
    reasons = retake_reasons(img, results, report)
    cv2.imwrite(out_path, sheet)
    counts = {k: sum(r["status"] == k for r in results) for k in ("right", "wrong", "review", "blank")}
    return json.dumps({"counts": counts, "tables": len({r["table"] for r in results}),
                       "retake": [msg for _, msg in reasons]})


def profile_check(path, out_path):
    """Debug: run check() under cProfile and return the slowest functions as text."""
    import cProfile
    import io
    import pstats
    pr = cProfile.Profile()
    pr.enable()
    result = check(path, out_path)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(28)
    return result + "\n" + s.getvalue()
