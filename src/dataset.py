from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class GateSample:
    image_path: Path
    mask_path: Path
    stem: str


def _as_hw(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("image_size must be an int or a 2-item sequence")
    return int(size[0]), int(size[1])


def read_split_file(split_file: Optional[Union[str, Path]]) -> Optional[set[str]]:
    if split_file is None:
        return None

    path = Path(split_file)
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")

    stems: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()[0]
        stems.add(Path(token).stem)

    if not stems:
        raise RuntimeError(f"Split file is empty: {path}")
    return stems


def collect_gate_samples(
    images_dir: Union[str, Path],
    masks_dir: Union[str, Path],
    split_file: Optional[Union[str, Path]] = None,
) -> List[GateSample]:
    images_dir = Path(images_dir)
    masks_dir = Path(masks_dir)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    allowed_stems = read_split_file(split_file)

    mask_map = {
        path.stem: path
        for path in masks_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MASK_EXTENSIONS
    }

    image_paths = sorted(
        [
            path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda p: p.stem,
    )

    samples: List[GateSample] = []
    for image_path in image_paths:
        if allowed_stems is not None and image_path.stem not in allowed_stems:
            continue
        mask_path = mask_map.get(image_path.stem)
        if mask_path is not None:
            samples.append(GateSample(image_path=image_path, mask_path=mask_path, stem=image_path.stem))

    return samples


class GateAugmenter:
    """Small synchronized augmentation set for gate binary segmentation."""

    def __init__(self, hflip: bool = False):
        self.hflip = hflip

    def __call__(self, image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < 0.75:
            image_rgb = self._brightness_contrast(image_rgb)
        if np.random.rand() < 0.35:
            image_rgb = self._gamma(image_rgb)
        if np.random.rand() < 0.30:
            image_rgb = self._blur(image_rgb)
        if np.random.rand() < 0.35:
            image_rgb = self._noise(image_rgb)
        if np.random.rand() < 0.75:
            image_rgb, mask = self._affine(image_rgb, mask)
        if np.random.rand() < 0.25:
            image_rgb, mask = self._perspective(image_rgb, mask)
        if self.hflip and np.random.rand() < 0.50:
            image_rgb = np.ascontiguousarray(image_rgb[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])

        mask = (mask > 0).astype(np.uint8)
        return image_rgb, mask

    @staticmethod
    def _brightness_contrast(image_rgb: np.ndarray) -> np.ndarray:
        alpha = np.random.uniform(0.75, 1.30)
        beta = np.random.uniform(-30.0, 30.0)
        return cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)

    @staticmethod
    def _gamma(image_rgb: np.ndarray) -> np.ndarray:
        gamma = np.random.uniform(0.70, 1.45)
        values = np.arange(256, dtype=np.float32) / 255.0
        table = np.clip((values ** gamma) * 255.0, 0, 255).astype(np.uint8)
        return cv2.LUT(image_rgb, table)

    @staticmethod
    def _blur(image_rgb: np.ndarray) -> np.ndarray:
        kernel = int(np.random.choice([3, 5]))
        return cv2.GaussianBlur(image_rgb, (kernel, kernel), sigmaX=0.0)

    @staticmethod
    def _noise(image_rgb: np.ndarray) -> np.ndarray:
        sigma = np.random.uniform(3.0, 12.0)
        noise = np.random.normal(0.0, sigma, image_rgb.shape).astype(np.float32)
        noisy = image_rgb.astype(np.float32) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    @staticmethod
    def _affine(image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = mask.shape[:2]
        angle = np.random.uniform(-8.0, 8.0)
        scale = np.random.uniform(0.88, 1.12)
        shift_x = np.random.uniform(-0.06, 0.06) * width
        shift_y = np.random.uniform(-0.06, 0.06) * height

        matrix = cv2.getRotationMatrix2D((width * 0.5, height * 0.5), angle, scale)
        matrix[0, 2] += shift_x
        matrix[1, 2] += shift_y

        image_rgb = cv2.warpAffine(
            image_rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpAffine(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return image_rgb, mask

    @staticmethod
    def _perspective(image_rgb: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        height, width = mask.shape[:2]
        jitter = 0.035 * min(height, width)
        src = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        dst = src + np.random.uniform(-jitter, jitter, src.shape).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)

        image_rgb = cv2.warpPerspective(
            image_rgb,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        mask = cv2.warpPerspective(
            mask,
            matrix,
            (width, height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return image_rgb, mask


class GateMaskDataset(Dataset):
    def __init__(
        self,
        images_dir: Union[str, Path],
        masks_dir: Union[str, Path],
        image_size: Union[int, Sequence[int]] = 384,
        split_file: Optional[Union[str, Path]] = None,
        augment: bool = False,
        hflip: bool = False,
        normalize: bool = True,
        return_meta: bool = False,
        augmenter: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.image_size = _as_hw(image_size)
        self.normalize = normalize
        self.return_meta = return_meta
        self.samples = collect_gate_samples(self.images_dir, self.masks_dir, split_file)

        if len(self.samples) == 0:
            split_hint = f" split_file={split_file}" if split_file else ""
            raise RuntimeError(
                f"No image-mask pairs found in {self.images_dir} and {self.masks_dir}.{split_hint} "
                "Mask files must share the same stem as image files."
            )

        if augmenter is not None:
            self.augmenter = augmenter
        elif augment:
            self.augmenter = GateAugmenter(hflip=hflip)
        else:
            self.augmenter = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        image_bgr = cv2.imread(str(sample.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {sample.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(sample.mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {sample.mask_path}")

        height, width = self.image_size
        image_rgb = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.uint8)

        if self.augmenter is not None:
            image_rgb, mask = self.augmenter(image_rgb, mask)

        image = image_rgb.astype(np.float32) / 255.0
        if self.normalize:
            image = (image - IMAGENET_MEAN) / IMAGENET_STD

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        image_tensor = torch.from_numpy(np.ascontiguousarray(image)).float()
        mask_tensor = torch.from_numpy(np.ascontiguousarray(mask)).float()

        if self.return_meta:
            meta = {
                "stem": sample.stem,
                "image_path": str(sample.image_path),
                "mask_path": str(sample.mask_path),
            }
            return image_tensor, mask_tensor, meta

        return image_tensor, mask_tensor
