# Abacus worksheet checker

Checks photos of abacus / mental-maths worksheets (SC-MAC style: columns of numbers, child writes the
answer in the ANS row) and marks every answer right, wrong, or "please check". Runs fully offline.

## How it works

1. **Find the tables** in the photo, fix rotation (sideways / upside down), and straighten them
   (camera tilt and curved book pages are handled by tracing each column line).
2. **Cut out cells**: the printed numbers in each column and the child's answer in the ANS row.
3. **Read every cell** with one small model (CRNN + CTC, ~1.4M parameters) that reads both printed
   numbers and handwriting: digits, `-`, `.`, `×`, `÷`.
4. **Work out the correct answer** in code, top to bottom (`+`, `−`, `×`, `÷`, decimals).
5. **Decide**:
   - reading matches the correct answer → right
   - the model is sure the child wrote something else, or the correct answer is implausible for the
     handwriting (e.g. wrong number of digits) → wrong
   - otherwise → sent to the teacher, with the expected answer shown

The correct answer never turns an unclear reading into a "right"; it only stops a doubtful "wrong".

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
| `reader.pt`, `reader_ft.pt` | Base and fine-tuned model weights |

Worksheet photos are not included in this repo (they contain children's handwriting).
`realset.py`, `finetune.py` and `e2e.py` expect them next to the `prototype/` folder.

## Status

Prototype. On held-out photos: printed numbers 99%+ read correctly; ~82% of answers checked
automatically, the rest sent to the teacher; no answer mis-marked so far. More handwriting samples
from different children are the main way to improve it.
