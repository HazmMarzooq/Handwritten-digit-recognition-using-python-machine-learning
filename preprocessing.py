"""Convert an uploaded image into the format of sklearn's digits dataset:
8x8 pixels, integer intensities 0..16, bright digit on a dark background."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps


@dataclass
class PreprocessResult:
    original: Image.Image
    gray: np.ndarray          # grayscale, original size (ink = bright)
    thresholded: np.ndarray   # background removed with Otsu threshold
    prepared: np.ndarray      # cropped + centred square image
    small: np.ndarray         # 8x8 float, 0..255 (after resize)
    matrix: np.ndarray        # 8x8 int, 0..16 (dataset scale)
    features: np.ndarray      # flattened, 64 values
    inverted: bool
    threshold: int


def load_image(file) -> Image.Image:
    img = ImageOps.exif_transpose(Image.open(file))
    if img.mode in ("RGBA", "LA", "P"):  # flatten transparency onto white
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        img = bg
    return img.convert("RGB")


def otsu_threshold(gray: np.ndarray) -> int:
    """Classic Otsu threshold (pure NumPy)."""
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    total = gray.size
    sum_all = float(np.dot(np.arange(256), hist))
    w_b = sum_b = best = 0.0
    thr = 0
    for t in range(256):
        w_b += hist[t]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += t * hist[t]
        var = w_b * w_f * (sum_b / w_b - (sum_all - sum_b) / w_f) ** 2
        if var > best:
            best, thr = var, t
    return thr


def preprocess_image(file, auto_invert: bool = True, crop: bool = True) -> PreprocessResult:
    original = load_image(file)

    # 1. Grayscale
    gray = np.array(original.convert("L"), dtype=np.uint8)

    # 2. The dataset has bright ink on a dark background. Photos/scans are
    #    usually dark ink on white paper, so invert when the border is bright.
    border = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    inverted = bool(auto_invert and np.median(border) > 127)
    if inverted:
        gray = 255 - gray

    # 3. Thresholding (Otsu): separate digit from background
    thr = otsu_threshold(gray)
    mask = gray > thr
    if not mask.any():
        raise ValueError("No digit found in the image (it looks blank).")
    thresholded = np.where(mask, gray, 0).astype(np.uint8)

    # 4. Crop to the digit and centre it on a square canvas with a margin,
    #    like the digits in the training set.
    if crop:
        ys, xs = np.where(mask)
        region = thresholded[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        pad_ratio = 0.17
    else:
        region = thresholded
        pad_ratio = 0.0
    h, w = region.shape
    side = max(h, w)
    pad = int(round(side * pad_ratio))
    canvas = np.zeros((side + 2 * pad, side + 2 * pad), dtype=np.uint8)
    top, left = pad + (side - h) // 2, pad + (side - w) // 2
    canvas[top:top + h, left:left + w] = region

    # 5. Resize to 8x8 (area averaging)
    small = np.array(
        Image.fromarray(canvas).resize((8, 8), Image.Resampling.BOX), dtype=float
    )
    small[small < 0.1 * small.max()] = 0  # remove faint residue

    # 6. Normalize to the dataset's 0..16 integer scale
    matrix = np.rint(small / small.max() * 16).astype(int)

    # 7. Flatten to 64 features
    features = matrix.flatten()

    return PreprocessResult(original, gray, thresholded, canvas, small,
                            matrix, features, inverted, thr)
