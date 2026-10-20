"""Shrink ONNX models for the phone: weights stored as 8-bit integers (dynamic quantization).

usage: python quantize_models.py model.onnx [...]   -> model_int8.onnx
Readings can change slightly, so re-run e2e.py / compare_models.py with the _int8 models before using them.
"""
import os
import sys
from onnxruntime.quantization import QuantType, quantize_dynamic
from onnxruntime.quantization.shape_inference import quant_pre_process

for path in sys.argv[1:]:
    stem = os.path.splitext(path)[0]
    prep, out = stem + "_prep.onnx", stem + "_int8.onnx"
    quant_pre_process(path, prep, skip_symbolic_shape=True)
    quantize_dynamic(prep, out, weight_type=QuantType.QUInt8)
    os.remove(prep)
    print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB -> {out}: {os.path.getsize(out) / 1e6:.1f} MB")
