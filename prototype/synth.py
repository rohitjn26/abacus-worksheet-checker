"""Synthetic training cells: handwritten numbers built from MNIST digits, and printed numbers from fonts.

Each sample is rendered as a photo-like grey crop and then goes through the same normalize() as real cells.
"""
import glob
import io
import random
import cv2
import numpy as np
import torchvision
from PIL import Image, ImageDraw, ImageFont
from norm import normalize

FONT_DIR = "/System/Library/Fonts/Supplemental/"
FONTS = [FONT_DIR + f for f in [
    "Arial.ttf", "Arial Bold.ttf", "Arial Narrow.ttf", "Times New Roman.ttf", "Times New Roman Bold.ttf",
    "Verdana.ttf", "Tahoma.ttf", "Trebuchet MS.ttf", "Georgia.ttf", "Courier New.ttf", "Courier New Bold.ttf",
]] + ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Geneva.ttf"]


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV"]


class Synth:
    def __init__(self, root="../data", split="train"):
        ds = torchvision.datasets.MNIST(root, train=(split == "train"), download=False)
        imgs, labels = ds.data.numpy(), ds.targets.numpy()
        self.by_digit = [imgs[labels == d] for d in range(10)]
        self.fonts = {}

    # ---------- labels ----------
    def number(self, printed):
        n_digits = random.choices([1, 2, 3, 4, 5, 6], [30, 35, 17, 9, 5, 4])[0]
        s = str(random.randint(1, 9)) + "".join(random.choice("0123456789") for _ in range(n_digits - 1))
        if random.random() < 0.05:
            s = "0"
        if random.random() < 0.1 and len(s) >= 2:
            k = random.randint(1, min(2, len(s) - 1))
            s = s[:-k] + "." + s[-k:]
        if random.random() < (0.3 if printed else 0.08):
            s = "-" + s
        if printed and random.random() < 0.15:
            s = random.choice("×÷+") + s.lstrip("-")
        return s

    # ---------- handwriting ----------
    def hand_glyph(self, ch, h):
        if ch.isdigit():
            g = random.choice(self.by_digit[int(ch)]).astype(np.float32) / 255
            ys, xs = np.nonzero(g > 0.2)
            g = g[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            gw = max(int(g.shape[1] * h / g.shape[0] * random.uniform(0.8, 1.2)), 2)
            g = cv2.resize(g, (gw, h))
            k = random.choice([0, 0, 1, 2, -1])
            if k > 0:
                g = cv2.dilate(g, np.ones((k + 1, k + 1), np.uint8))
            elif k < 0:
                g = cv2.erode(g, np.ones((2, 2), np.uint8))
            return g, 0
        if ch == "-":
            L, t = random.randint(int(h * 0.3), int(h * 0.6)), random.randint(2, 5)
            g = np.zeros((t + 6, L + 4), np.float32)
            cv2.line(g, (2, 3 + random.randint(-2, 2)), (L + 1, 3 + random.randint(-2, 2)), 1.0, t)
            return g, h // 2 - g.shape[0] // 2 + random.randint(-4, 4)
        r = random.randint(2, 4)  # '.'
        g = np.zeros((2 * r + 3, 2 * r + 3), np.float32)
        cv2.circle(g, (r + 1, r + 1), r, 1.0, -1)
        return g, h - g.shape[0] + random.randint(-2, 2)

    def handwritten(self, text):
        h = random.randint(30, 46)
        glyphs = [self.hand_glyph(c, h) for c in text]
        W = sum(g.shape[1] for g, _ in glyphs) + 12 * len(glyphs) + 40
        canvas = np.zeros((h + 40, W), np.float32)
        x = 20
        for g, dy in glyphs:
            y = 20 + dy + random.randint(-4, 4)
            gh, gw = g.shape
            y = min(max(y, 0), canvas.shape[0] - gh)
            canvas[y : y + gh, x : x + gw] = np.maximum(canvas[y : y + gh, x : x + gw], g)
            x += gw + random.randint(-6, 10)
            x = max(x, 0)
        canvas = canvas[:, : x + 20]
        if random.random() < 0.7:  # slant
            sh = random.uniform(-0.35, 0.35)
            M = np.float32([[1, sh, -sh * canvas.shape[0] / 2], [0, 1, 0]])
            canvas = cv2.warpAffine(canvas, M, (canvas.shape[1], canvas.shape[0]))
        return canvas, random.uniform(0.3, 0.85)  # pencil is lighter than print

    # ---------- print ----------
    def printed(self, text):
        size = random.randint(28, 44)
        key = (random.choice(FONTS), size)
        if key not in self.fonts:
            self.fonts[key] = ImageFont.truetype(*key)
        font = self.fonts[key]
        if random.random() < 0.5:
            text_r = text.replace("×", random.choice(["×", "x", "X"]))
        else:
            text_r = text
        if text_r[:1] in "×÷+x" and random.random() < 0.5:
            text_r = text_r[0] + " " + text_r[1:]
        l, t, r, b = font.getbbox(text_r)
        img = Image.new("L", (r - l + 40, b - t + 40), 0)
        ImageDraw.Draw(img).text((20 - l, 20 - t), text_r, fill=255, font=font)
        return np.asarray(img, np.float32) / 255, random.uniform(0.55, 0.95)

    # ---------- photo look ----------
    def photo(self, ink, darkness):
        h, w = ink.shape
        if random.random() < 0.5:  # clutter: leftover grid-line bits and neighbouring marks at the edges
            for _ in range(random.randint(1, 3)):
                if random.random() < 0.5:
                    y = random.choice([random.randint(0, 6), h - random.randint(1, 7)])
                    cv2.line(ink, (0, y), (w, y + random.randint(-2, 2)), random.uniform(0.2, 0.6), random.randint(1, 2))
                else:
                    x = random.choice([random.randint(0, 5), w - random.randint(1, 6)])
                    cv2.line(ink, (x, 0), (x, h), random.uniform(0.2, 0.6), random.randint(1, 2))
        bg = random.uniform(110, 225)
        grad = np.linspace(0, random.uniform(-25, 25), w)[None, :]
        img = bg + grad - ink * darkness * bg
        img += np.random.normal(0, random.uniform(1, 7), img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        if random.random() < 0.6:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.4, 1.3))
        s = random.uniform(0.45, 1.0)  # phone photos are low-res
        img = cv2.resize(img, (max(int(w * s), 8), max(int(h * s), 8)), interpolation=cv2.INTER_AREA)
        if random.random() < 0.7:
            _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(30, 90)])
            img = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)
        return img

    def sample(self):
        r = random.random()
        if r < 0.03:
            text, ink, dark = "", np.zeros((60, 120), np.float32), 0.5
        else:
            printed = r < 0.45
            text = self.number(printed)
            if printed and random.random() < 0.1:
                text = random.choice(ROMAN)
            ink, dark = self.printed(text) if printed else self.handwritten(text)
        return normalize(self.photo(ink, dark)), text
