"""Export PaddleOCR PP-OCRv5 mobile's frozen part (backbone + CTC neck) to ONNX, and check it matches."""
import sys
import numpy as np
import paddle
from paddle.static import InputSpec

sys.argv = ["x"]
src = open("extract_feats.py").read().split("out = {}")[0]
exec(src)  # builds `model` with pretrained weights, and `prep`


class Frozen(paddle.nn.Layer):
    def __init__(self, m):
        super().__init__()
        self.backbone, self.neck = m.backbone, m.head.ctc_encoder

    def forward(self, x):
        return self.neck(self.backbone(x))


net = Frozen(model)
net.eval()
static = paddle.jit.to_static(net, input_spec=[InputSpec([None, 3, 48, 320], "float32", "x")], full_graph=True)
paddle.onnx.export(static, "ppocrv5_mobile_frozen", input_spec=[InputSpec([None, 3, 48, 320], "float32", "x")],
                   opset_version=14)

import onnxruntime as ort  # noqa: E402
rows = [l.split("\t") for l in open("cells/test.txt").read().splitlines()][:32]
x = np.stack([prep("cells/" + r[0]) for r in rows])
with paddle.no_grad():
    ref = net(paddle.to_tensor(x)).numpy()
out = ort.InferenceSession("ppocrv5_mobile_frozen.onnx").run(None, {"x": x})[0]
print("output shape", out.shape, " max abs diff vs Paddle:", float(np.abs(out - ref).max()))
