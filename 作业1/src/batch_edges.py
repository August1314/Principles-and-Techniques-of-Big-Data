import argparse
from pathlib import Path

import cv2

from edge_sobel import make_demo_image, sobel_edges


def process_one(image_path: Path, outdir: Path, prefix: str, ksize: int) -> None:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    edges = sobel_edges(bgr, ksize=ksize)
    cv2.imwrite(str(outdir / f"{prefix}_original.png"), bgr)
    cv2.imwrite(str(outdir / f"{prefix}_sobel.png"), edges)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--ksize", type=int, default=3)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Demo image
    demo_path = root / "data" / "edge_demo.png"
    make_demo_image(demo_path, width=720, height=460)
    process_one(demo_path, outdir, "edge1_demo", ksize=args.ksize)

    # 2) Slide photo (assignment image) if present
    slide = root / "作业1说明.jpg"
    if slide.exists():
        process_one(slide, outdir, "edge2_slide", ksize=args.ksize)

    # 3) Blurred version of demo
    bgr = cv2.imread(str(demo_path), cv2.IMREAD_COLOR)
    blurred = cv2.GaussianBlur(bgr, (11, 11), 0)
    blur_path = root / "data" / "edge_demo_blur.png"
    cv2.imwrite(str(blur_path), blurred)
    process_one(blur_path, outdir, "edge3_blur", ksize=args.ksize)

    # 4) Noisy version of demo
    noise = (cv2.randn(bgr.copy(), (0, 0, 0), (18, 18, 18))).astype("uint8")
    noisy = cv2.add(bgr, noise)
    noisy_path = root / "data" / "edge_demo_noisy.png"
    cv2.imwrite(str(noisy_path), noisy)
    process_one(noisy_path, outdir, "edge4_noisy", ksize=args.ksize)

    print(f"Saved edge examples to: {outdir}")


if __name__ == "__main__":
    main()

