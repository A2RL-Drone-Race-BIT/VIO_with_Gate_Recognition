from __future__ import annotations

import argparse
import random
from pathlib import Path

from dataset import collect_gate_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create reproducible train/val split files.")
    parser.add_argument("--images", type=str, default="data/images")
    parser.add_argument("--masks", type=str, default="data/masks")
    parser.add_argument("--out-dir", type=str, default="data/splits")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = collect_gate_samples(args.images, args.masks)
    if len(samples) < 2:
        raise RuntimeError("Need at least 2 image-mask pairs to create a split.")

    rng = random.Random(args.seed)
    stems = [sample.stem for sample in samples]
    rng.shuffle(stems)

    val_len = max(1, int(round(len(stems) * args.val_ratio)))
    val_len = min(val_len, len(stems) - 1)
    val_stems = sorted(stems[:val_len])
    train_stems = sorted(stems[val_len:])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "train.txt").write_text("\n".join(train_stems) + "\n", encoding="utf-8")
    (out_dir / "val.txt").write_text("\n".join(val_stems) + "\n", encoding="utf-8")

    print(f"Pairs: {len(stems)}")
    print(f"Train: {len(train_stems)} -> {out_dir / 'train.txt'}")
    print(f"Val:   {len(val_stems)} -> {out_dir / 'val.txt'}")


if __name__ == "__main__":
    main()
