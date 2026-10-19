# Abacus worksheet checker

Checks photos of abacus / mental-maths worksheets (SC-MAC style: columns of numbers, child writes the
answer in the ANS row) and marks every answer right, wrong, or "please check". Runs fully offline.

## How it works

1. **Find the tables** in the photo, fix rotation (sideways / upside down), and straighten them
   (camera tilt and curved book pages are handled by tracing each column line). The photo counts as
   upright when the model can read the `ANS` label; otherwise it is turned 180°.
2. **Cut out cells**: the printed numbers in each column and the child's answer in the ANS row.
3. **Read every cell**:
   - printed numbers, column headers (`I V X`) and the label column (`A N S B C`): our own small model
     (CRNN + CTC, ~1.4M parameters, 5 MB)
   - the child's handwritten answer: PaddleOCR's PP-OCRv5 mobile recognizer with its pretrained layers
     frozen (run via ONNX, 22 MB) plus a small head trained on our cells. Pretraining on millions of
     real text images reads children's handwriting better than our model (held-out cells: ~88% vs 81%).

   Both run with ONNX / PyTorch only, so the pipeline can move to Android / iOS.
4. **Work out the correct answer** in code, top to bottom (`+`, `−`, `×`, `÷`, decimals).
5. **Decide**:
   - reading matches the correct answer → right
   - the model is sure the child wrote something else, or the correct answer is implausible for the
     handwriting (e.g. wrong number of digits) → wrong
   - otherwise → sent to the teacher, with the expected answer shown

The correct answer never turns an unclear reading into a "right"; it only stops a doubtful "wrong".

## Photo-quality check (retake requests)

Before marking, `quality.retake_reasons` decides whether the parent should retake the photo: too dark,
grainy, blurry, glare, strong shadow, too far away, numbers unclear, worksheet not found, or part of the
sheet unreadable. Thresholds come from `degrade_test.py`, which makes the sample photos progressively
darker / blurrier / glarier / smaller etc. and finds where wrong marks start. On the held-out photos,
across all degraded versions, the check accepts no photo that gets a wrong mark, and never asks to retake
a good photo.

## Training data

- Synthetic cells (MNIST digits + fonts) - `synth.py`
- Real cells from the training photos - `realset.py`
- Whole-photo augmentation: random dark / shadow / glare / blur / shake / noise / distance / JPEG / tilt
  versions of the training photos, labelled from the transcription - `augment_photos.py`
  (held-out photos are never augmented or trained on).

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cd prototype
../.venv/bin/python check.py "../worksheet photo.jpeg"
```

Marked sheets are written to `prototype/results/`:
`OK` = right, `X n` = wrong (n = correct answer), `? = n` = please check (expected answer n),
`*` = printed numbers unclear, verify the sum.

## Files (`prototype/`)

| File | What it does |
|---|---|
| `check.py` | End-to-end checker (entry point) |
| `grid.py` | Orientation, table finding, column tracing and straightening |
| `cells.py` | Row finding, cell cut-outs, answer crops |
| `norm.py` | Cell clean-up and normalisation for the model |
| `model.py` | CRNN reader, decoding, string probabilities |
| `policy.py` | Right / wrong / review rule |
| `synth.py`, `train.py` | Synthetic training data (MNIST digits + fonts) and base training |
| `realset.py`, `finetune.py` | Real labelled cells and fine-tuning |
| `calibrate.py`, `e2e.py` | Accuracy measurement on held-out photos |
| `testset.json` | Hand transcription of the sample worksheets |
| `reader_v4.pt` | Our model: printed numbers, headers, ANS label (reads "ANS", "A N S", "Ans", "ans") |
| `hand_reader.py`, `hand_head_v2.pt`, `ppocrv5_mobile_frozen.onnx` | Handwriting reader: frozen PaddleOCR + trained head |
| `quality.py`, `degrade_test.py` | Photo-quality check (retake requests) and the bad-photo stress test |
| `augment_photos.py` | Whole-photo augmentation of the training photos |
| `train_hand_head.py` | Trains the handwriting head on frozen PaddleOCR features |
| `reader.pt` … `reader_v3.pt`, `hand_head.pt` | Earlier weights, kept as training starting points |
| `run_folder.py`, `prefill.py` | Check a whole folder; pre-fill a transcription from the checker's readings |
| `compare_models.py` | Compare model combinations end to end on the held-out photos |

Worksheet photos are not included in this repo (they contain children's handwriting).
`realset.py`, `finetune.py` and `e2e.py` expect them next to the `prototype/` folder.

## PaddleOCR experiment (`paddle_exp/`)

Scripts used to compare and build the handwriting reader (Python 3.12, `requirements-paddle.txt`):
`zeroshot.py` (PaddleOCR as-is), `extract_feats.py` (frozen features), `train_last_layer.py`
(last-layer comparison), `export_onnx.py` (frozen part to ONNX). Data, the cloned PaddleOCR repo and
weights are not committed.

## Third-party

PP-OCRv5 mobile recognizer weights © PaddlePaddle Authors, Apache License 2.0
(https://github.com/PaddlePaddle/PaddleOCR). `prototype/ppocrv5_mobile_frozen.onnx` is a converted
subset of those weights.

## Status

Prototype. 25 transcribed photos (512 answers, 3,325 printed numbers); 10 photos held out for testing.
On the held-out photos: printed numbers 99%+ read correctly; ~88% of answers checked automatically, the
rest sent to the teacher; no answer mis-marked so far, including on artificially degraded versions of
those photos that pass the quality check.

Calibration (`calibrate_head.py`): three training photos are kept apart (`CALIB_PHOTOS` in `realset.py`)
to tune a temperature that makes the handwriting head's confidence honest. A stronger head trained this
way checks 91% of held-out answers automatically but still made one wrong mark (a very faint "14" read
as "4"), so the default stays the earlier head (88%, no wrong marks). With more sheets - in particular a
larger calibration set - re-run: `build_hand_feats.py`, `augment_photos.py`, `train_hand_head.py`,
`calibrate_head.py`, then `compare_models.py --calib` and finally `compare_models.py`. More handwriting samples
from different children are the main way to improve it.
