"""Run PaddleOCR PP-OCRv5 mobile's pretrained layers (frozen) over every cell once and save the features
that feed its last layer. Then only that last layer is trained (train_last_layer.py)."""
import sys
import cv2
import numpy as np
import paddle
import yaml

sys.path.insert(0, "PaddleOCR")
from ppocr.modeling.architectures import build_model  # noqa: E402
from ppocr.utils.save_load import load_model  # noqa: E402

cfg = yaml.safe_load(open("PaddleOCR/configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"))
n_chars = len(open("PaddleOCR/ppocr/utils/dict/ppocrv5_dict.txt", encoding="utf-8").read().splitlines()) + 2
cfg["Architecture"]["Head"]["out_channels_list"] = {"CTCLabelDecode": n_chars, "SARLabelDecode": n_chars + 2,
                                                    "NRTRLabelDecode": n_chars + 3}
model = build_model(cfg["Architecture"])
cfg["Global"]["pretrained_model"] = "./PP-OCRv5_mobile_rec_pretrained"
load_model(cfg, model, model_type="rec")
model.eval()


def prep(path, H=48, W=320):
    """PP-OCR recognizer preprocessing: keep aspect, height 48, pad right to width 320, scale to [-1, 1]."""
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    rw = min(W, int(np.ceil(H * w / h)))
    x = (cv2.resize(img, (rw, H)).astype(np.float32) / 255 - 0.5) / 0.5
    out = np.zeros((H, W, 3), np.float32)
    out[:, :rw] = x
    return out.transpose(2, 0, 1)


def run(list_file, n=None):
    rows = [l.split("\t") for l in open(list_file).read().splitlines()][:n]
    feats = []
    with paddle.no_grad():
        for i in range(0, len(rows), 64):
            x = paddle.to_tensor(np.stack([prep("cells/" + r[0]) for r in rows[i : i + 64]]))
            feats.append(model.head.ctc_encoder(model.backbone(x)).numpy().astype(np.float16))
    kinds = [r[2] if len(r) > 2 else "synth" for r in rows]
    return np.concatenate(feats), [r[1] for r in rows], kinds


out = {}
for split, lst, n in [("train", "cells/train.txt", None), ("synth", "cells/synth.txt", 6000), ("test", "cells/test.txt", None)]:
    f, y, k = run(lst, n)
    out[f"{split}_x"], out[f"{split}_y"], out[f"{split}_kind"] = f, np.array(y), np.array(k)
    print(split, f.shape, flush=True)
np.savez_compressed("paddle_feats.npz", **out)
