"""Handwritten-answer reader: PaddleOCR PP-OCRv5 mobile (frozen, run via ONNX) + a small trained head.

PaddleOCR's pretrained layers turn a cell image into 40 x 120 features; only the head (BiLSTM + Linear)
was trained on our cells. Same interface as check.Reader.log_probs: CTC log-probs (T, B, C).
PP-OCRv5 weights: Apache-2.0, https://github.com/PaddlePaddle/PaddleOCR
"""
import json
import os
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

HAND_ALPHABET = "0123456789-.×÷+"  # CTC blank is index 0


class HandHead(nn.Module):
    def __init__(self, n_chars=len(HAND_ALPHABET), layers=1):
        super().__init__()
        self.rnn = nn.LSTM(120, 96, num_layers=layers, bidirectional=True, batch_first=True,
                           dropout=0.1 if layers > 1 else 0.0)
        self.fc = nn.Linear(192, n_chars + 1)

    def forward(self, feats):  # (B, 40, 120) -> logits (B, 40, C)
        return self.fc(self.rnn(feats)[0])


def prep(gray, H=48, W=320):
    """PP-OCR recognizer input: keep aspect, height 48, pad right to width 320, scale to [-1, 1]."""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    h, w = img.shape[:2]
    rw = min(W, int(np.ceil(H * w / h)))
    x = (cv2.resize(img, (rw, H)).astype(np.float32) / 255 - 0.5) / 0.5
    out = np.zeros((H, W, 3), np.float32)
    out[:, :rw] = x
    return out.transpose(2, 0, 1)


class HandReader:
    def __init__(self, onnx_path="ppocrv5_mobile_frozen.onnx", head_path="hand_head.pt", alphabet=HAND_ALPHABET):
        self.alphabet = alphabet
        self.session = ort.InferenceSession(onnx_path)
        state = torch.load(head_path, map_location="cpu")
        self.head = HandHead(len(alphabet), layers=2 if "rnn.weight_ih_l1" in state else 1)
        self.head.load_state_dict(state)
        self.head.eval()
        # Confidence calibration (calibrate_head.py): logits are divided by this temperature.
        cal = os.path.splitext(head_path)[0] + ".json"
        self.temperature = json.load(open(cal))["temperature"] if os.path.exists(cal) else 1.0

    def features(self, crops):
        return self.session.run(None, {"x": np.stack([prep(c) for c in crops])})[0]

    def log_probs(self, crops):
        with torch.no_grad():
            logits = self.head(torch.from_numpy(self.features(crops)))
        return (logits / self.temperature).log_softmax(-1).permute(1, 0, 2)  # (T, B, C)

