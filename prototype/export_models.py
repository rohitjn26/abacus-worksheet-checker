"""Export trained PyTorch models to ONNX (what the checker and the Android app run).

usage: python export_models.py reader_v4.pt hand_head_v2.pt ...
  reader_*.pt  -> reader_*.onnx      input x (B, 1, 32, 128)  -> log_probs (T=32, B, C)
  hand_head_*.pt -> hand_head_*.onnx input feats (B, 40, 120) -> logits (B, 40, C)
A hand head's calibration file (<name>.json, temperature) is used as is by the ONNX runner.
"""
import sys
import numpy as np
import onnxruntime as ort
import torch
from hand_reader import HandHead
from model import load_reader

for path in sys.argv[1:]:
    out = path.rsplit(".", 1)[0] + ".onnx"
    if path.startswith("hand_head"):
        state = torch.load(path, map_location="cpu")
        model = HandHead(layers=2 if "rnn.weight_ih_l1" in state else 1)
        model.load_state_dict(state)
        x, names = torch.randn(3, 40, 120), (["feats"], ["logits"])
        axes = {"feats": {0: "B"}, "logits": {0: "B"}}
    else:
        model = load_reader(path)
        x, names = torch.rand(3, 1, 32, 128), (["x"], ["log_probs"])
        axes = {"x": {0: "B"}, "log_probs": {1: "B"}}
    model.eval()
    torch.onnx.export(model, x, out, input_names=names[0], output_names=names[1], dynamic_axes=axes,
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = model(x).numpy()
    got = ort.InferenceSession(out).run(None, {names[0][0]: x.numpy()})[0]
    print(f"{path} -> {out}: max diff vs PyTorch {float(np.abs(ref - got).max()):.2e}")
