from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def threshold_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    if mask.dtype != np.uint8:
        mask = np.clip(mask, 0.0, 1.0)
        mask = (mask * 255.0).astype(np.uint8)

    threshold_value = int(threshold * 255) if threshold <= 1.0 else int(threshold)
    _, binary = cv2.threshold(mask, threshold_value, 255, cv2.THRESH_BINARY)
    return binary


def clean_mask(binary: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size <= 1:
        return binary

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return binary


def order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] != 4:
        raise ValueError("order_corners expects exactly four points")

    y_sorted = points[np.argsort(points[:, 1])]
    top = y_sorted[:2]
    bottom = y_sorted[2:]
    top = top[np.argsort(top[:, 0])]
    bottom = bottom[np.argsort(bottom[:, 0])]
    tl, tr = top
    bl, br = bottom
    return np.array([tl, tr, br, bl], dtype=np.float32)


def contour_to_quad(contour: np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(contour, closed=True)
    for epsilon_ratio in (0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, closed=True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_corners(box)


def extract_gate_corners(
    mask: np.ndarray,
    threshold: float = 0.5,
    kernel_size: int = 5,
    min_area: float = 100.0,
) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
    binary = threshold_mask(mask, threshold=threshold)
    binary = clean_mask(binary, kernel_size=kernel_size)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    debug: Dict[str, np.ndarray] = {"binary": binary}
    if not contours:
        return None, debug

    contour = max(contours, key=cv2.contourArea)
    debug["contour"] = contour
    corners = contour_to_quad(contour)
    return corners, debug


def draw_corners(
    image_bgr: np.ndarray,
    contour: Optional[np.ndarray],
    corners: Optional[np.ndarray],
) -> np.ndarray:
    vis = image_bgr.copy()
    if contour is not None:
        cv2.drawContours(vis, [contour], -1, (0, 255, 255), 2)

    if corners is not None:
        labels = ["tl", "tr", "br", "bl"]
        for label, point in zip(labels, corners):
            x, y = int(round(point[0])), int(round(point[1]))
            cv2.circle(vis, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(vis, label, (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
    return vis


def save_corner_outputs(
    out_dir: Path,
    stem: str,
    image_bgr: np.ndarray,
    binary_mask: np.ndarray,
    contour: Optional[np.ndarray],
    corners: Optional[np.ndarray],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = draw_corners(image_bgr, contour=contour, corners=corners)

    payload = {
        "found": corners is not None,
        "order": ["top_left", "top_right", "bottom_right", "bottom_left"],
        "corners": [] if corners is None else corners.round(2).tolist(),
    }

    cv2.imwrite(str(out_dir / f"{stem}_clean_mask.png"), binary_mask)
    cv2.imwrite(str(out_dir / f"{stem}_corners.png"), vis)
    (out_dir / f"{stem}_corners.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a gate quadrilateral from a predicted mask.")
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--min-area", type=float, default=100.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask = cv2.imread(args.mask, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {args.mask}")

    if args.image is not None:
        image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {args.image}")
    else:
        gray = threshold_mask(mask, threshold=args.threshold)
        image_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    corners, debug = extract_gate_corners(
        mask,
        threshold=args.threshold,
        kernel_size=args.kernel_size,
        min_area=args.min_area,
    )

    out_dir = Path(args.out_dir)
    stem = Path(args.mask).stem
    save_corner_outputs(
        out_dir=out_dir,
        stem=stem,
        image_bgr=image_bgr,
        binary_mask=debug["binary"],
        contour=debug.get("contour"),
        corners=corners,
    )
    print(f"Saved corner outputs to: {out_dir}")


if __name__ == "__main__":
    main()
