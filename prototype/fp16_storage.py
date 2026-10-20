"""Store an ONNX model's weights as float16 (half the file size) but compute in float32: each large weight
is saved in float16 followed by a Cast to float32, which ONNX Runtime folds away when it loads the model.

usage: python fp16_storage.py model.onnx  -> model_fp16w.onnx
"""
import os
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper

path = sys.argv[1]
out = os.path.splitext(path)[0] + "_fp16w.onnx"
m = onnx.load(path)
g = m.graph
keep, casts = [], []
for init in g.initializer:
    arr = numpy_helper.to_array(init)
    if arr.dtype == np.float32 and arr.size >= 1024:
        half = numpy_helper.from_array(arr.astype(np.float16), init.name + "__fp16")
        keep.append(half)
        casts.append(helper.make_node("Cast", [half.name], [init.name], to=onnx.TensorProto.FLOAT))
    else:
        keep.append(init)
del g.initializer[:]
g.initializer.extend(keep)
nodes = list(g.node)
del g.node[:]
g.node.extend(casts + nodes)
onnx.save(m, out)
print(f"{path}: {os.path.getsize(path) / 1e6:.1f} MB -> {out}: {os.path.getsize(out) / 1e6:.1f} MB")
