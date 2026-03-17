import argparse
import os
from pathlib import Path

import cv2
import numpy as np


def make_demo_image(out_path: Path, width: int = 640, height: int = 420) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (245, 245, 245)

    cv2.rectangle(img, (40, 40), (260, 200), (30, 30, 30), 3)
    cv2.circle(img, (470, 130), 70, (30, 30, 30), 3)
    cv2.line(img, (40, 320), (600, 320), (30, 30, 30), 5)
    cv2.putText(
        img,
        "Sobel Demo",
        (60, 285),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.3,
        (30, 30, 30),
        3,
        cv2.LINE_AA,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def sobel_edges(bgr: np.ndarray, ksize: int = 3) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)

    magnitude_u8 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    _, edges = cv2.threshold(
        magnitude_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return edges


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="data/edge_input.png")
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--ksize", type=int, default=3)
    parser.add_argument("--make-demo", type=str, default=None)
    args = parser.parse_args()

    if args.make_demo:
        make_demo_image(Path(args.make_demo))
        print(f"Demo image written to: {args.make_demo}")
        return

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}. Use --make-demo to generate one."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    edges = sobel_edges(bgr, ksize=args.ksize)

    original_out = outdir / "edge_original.png"
    sobel_out = outdir / "edge_sobel.png"
    cv2.imwrite(str(original_out), bgr)
    cv2.imwrite(str(sobel_out), edges)

    print(f"Saved: {original_out}")
    print(f"Saved: {sobel_out}")


if __name__ == "__main__":
    main()

