"""Generate a reproducible synthetic sample photo for trying the CLI.

The image has Haar-detectable faces on a textured background, so
`facekeep compress` exercises the face-aware path (4:4:4 chroma on faces),
not just a plain encode. Deterministic: the same seed produces the same
pixels every run, so the sample never needs to be committed by hand.

Run:  python examples/make_sample.py
      facekeep compress examples/sample_family.jpg      # -> .avif

Colors are BGR (OpenCV convention), matching the rest of the project.
"""

from pathlib import Path

import cv2
import numpy as np


def draw_face(img: np.ndarray, cx: int, cy: int, fw: int) -> None:
    """Draw one simple Haar-detectable face (same style as the test fixtures)."""
    fh = int(fw * 1.3)
    cv2.ellipse(img, (cx, cy), (fw // 2, fh // 2), 0, 0, 360, (180, 170, 165), -1)
    cv2.ellipse(img, (cx, cy - fh // 6), (fw // 2 - 5, fh // 4), 0, 0, 360,
                (195, 185, 180), -1)
    eye_y = cy - fh // 10
    ew = fw // 7
    cv2.ellipse(img, (cx - fw // 5, eye_y), (ew, ew // 2), 0, 0, 360, (60, 55, 55), -1)
    cv2.ellipse(img, (cx + fw // 5, eye_y), (ew, ew // 2), 0, 0, 360, (60, 55, 55), -1)
    cv2.line(img, (cx, eye_y), (cx, cy + fh // 12), (150, 140, 135), max(2, fw // 40))
    cv2.ellipse(img, (cx, cy + fh // 4), (fw // 5, fh // 18), 0, 0, 180, (120, 90, 90), -1)


def make_sample(path: Path) -> Path:
    """Write a synthetic 'family photo' (two faces, textured background)."""
    rng = np.random.default_rng(3)
    H, W = 1200, 1800
    bg = cv2.resize(
        rng.normal(128, 30, (H // 10, W // 10, 3)).astype(np.float32),
        (W, H), interpolation=cv2.INTER_CUBIC,
    )
    img = np.clip(bg, 0, 255).astype(np.uint8)

    # Two faces at scales the bundled Haar cascade reliably detects.
    draw_face(img, 600, 580, 300)
    draw_face(img, 1250, 560, 280)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return path


if __name__ == "__main__":
    out = make_sample(Path(__file__).with_name("sample_family.jpg"))
    print(f"wrote {out}")
