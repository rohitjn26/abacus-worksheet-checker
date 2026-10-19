"""Runs the checker's ONNX models. No PyTorch, so the same code runs on a Mac and inside the Android app.

Backends: OrtBackend (onnxruntime, desktop) or any object with run(model_file, input_name, array) -> array,
e.g. the Android app's Kotlin bridge.
"""
import json
import os
import cv2
import numpy as np
from ctc import log_softmax

HERE = os.path.dirname(os.path.abspath(__file__))
HAND_ALPHABET = "0123456789-.×÷+"


class OrtBackend:
    def __init__(self):
        import onnxruntime as ort
        self._ort, self._sessions = ort, {}

    def run(self, model_file, input_name, x):
        if model_file not in self._sessions:
            self._sessions[model_file] = self._ort.InferenceSession(os.path.join(HERE, model_file))
        return self._sessions[model_file].run(None, {input_name: x})[0]


def paddle_prep(gray, H=48, W=320):
    """PP-OCR recognizer input: keep aspect, height 48, pad right to width 320, scale to [-1, 1]."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    h, w = img.shape[:2]
    rw = min(W, int(np.ceil(H * w / h)))
    x = (cv2.resize(img, (rw, H)).astype(np.float32) / 255 - 0.5) / 0.5
    out = np.zeros((H, W, 3), np.float32)
    out[:, :rw] = x
    return out.transpose(2, 0, 1)


def onnx_name(path):
    return os.path.splitext(path)[0] + ".onnx"


class Models:
    def __init__(self, reader="reader_v4.onnx", hand_head="hand_head_v2.onnx",
                 paddle="ppocrv5_mobile_frozen.onnx", backend=None):
        self.reader, self.hand_head, self.paddle = onnx_name(reader), onnx_name(hand_head), paddle
        self.backend = backend or OrtBackend()
        cal = os.path.join(HERE, os.path.splitext(self.hand_head)[0] + ".json")  # calibration (temperature)
        self.temperature = json.load(open(cal))["temperature"] if os.path.exists(cal) else 1.0

    def reader_log_probs(self, x):
        """x (B, 1, 32, 128) normalized cells -> log-probs (T, B, C) over norm.ALPHABET."""
        return self.backend.run(self.reader, "x", np.ascontiguousarray(x, np.float32))

    def hand_log_probs(self, crops):
        """Raw grey answer crops -> log-probs (T, B, C) over HAND_ALPHABET."""
        x = np.stack([paddle_prep(c) for c in crops]).astype(np.float32)
        feats = self.backend.run(self.paddle, "x", x)
        logits = self.backend.run(self.hand_head, "feats", np.ascontiguousarray(feats, np.float32))
        return log_softmax(logits / self.temperature).transpose(1, 0, 2)
