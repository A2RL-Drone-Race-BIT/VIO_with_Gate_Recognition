from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD
from model import MobileNetV3UNet


def preprocess(image_bgr: np.ndarray, size: int) -> torch.Tensor:
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    image = image_rgb.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(np.ascontiguousarray(image)).unsqueeze(0).float()


def make_overlay(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    overlay = image_bgr.copy()
    red = np.zeros_like(image_bgr)
    red[:, :, 2] = 255
    active = mask > 0
    overlay[active] = cv2.addWeighted(image_bgr, 1.0 - alpha, red, alpha, 0.0)[active]
    return overlay


@torch.no_grad()
def predict_probability(
    model: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    size: int,
) -> np.ndarray:
    orig_h, orig_w = image_bgr.shape[:2]
    x = preprocess(image_bgr, size=size).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    return cv2.resize(prob, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = MobileNetV3UNet(pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one-image gate mask inference.")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="checkpoints/best.pth")
    parser.add_argument("--out-dir", type=str, default="outputs")
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--postprocess", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")

    model = load_model(args.ckpt, device)
    prob = predict_probability(model, image_bgr, device=device, size=args.size)
    mask = (prob >= args.threshold).astype(np.uint8) * 255

    stem = Path(args.image).stem
    prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    overlay = make_overlay(image_bgr, mask)

    cv2.imwrite(str(out_dir / f"{stem}_prob.png"), prob_u8)
    cv2.imwrite(str(out_dir / f"{stem}_heat.png"), heat)
    cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)
    cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    if args.postprocess:
        from postprocess import extract_gate_corners, save_corner_outputs

        corners, debug = extract_gate_corners(mask, threshold=127, kernel_size=5)
        save_corner_outputs(
            out_dir=out_dir,
            stem=stem,
            image_bgr=image_bgr,
            binary_mask=debug["binary"],
            contour=debug.get("contour"),
            corners=corners,
        )

    print(f"Saved inference outputs to: {out_dir}")


if __name__ == "__main__":
    main()
