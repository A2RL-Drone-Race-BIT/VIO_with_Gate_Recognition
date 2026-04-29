# A2RL Gate Mask Net

Small binary segmentation workspace for the A2RL gate perception prototype.

The current pipeline is:

```text
RGB image -> MobileNetV3-UNet -> gate mask -> contour cleanup -> 4-corner visualization
```

The network takes a 384x384 RGB image and predicts a 384x384 single-channel gate mask.

## Layout

```text
a2rl_mask_net/
  data/
    images/        # raw gate images
    masks/         # binary masks with the same file stem as images
    splits/        # train.txt and val.txt
  src/
    dataset.py
    model.py
    train.py
    infer_one.py
    postprocess.py
    split_data.py
  checkpoints/
  outputs/
```

Example pair:

```text
data/images/frame_0001.jpg
data/masks/frame_0001.png
```

## Setup

```powershell
pip install -r requirements.txt
```

For CUDA, install the PyTorch wheel that matches your local CUDA version from the official PyTorch instructions.

## Create Splits

```powershell
python src\split_data.py --images data\images --masks data\masks --val-ratio 0.2
```

This writes:

```text
data/splits/train.txt
data/splits/val.txt
```

## Train

CPU or auto device:

```powershell
python src\train.py --images data\images --masks data\masks --train-split data\splits\train.txt --val-split data\splits\val.txt --epochs 80 --batch-size 8 --size 384
```

With pretrained MobileNetV3 weights:

```powershell
python src\train.py --pretrained --auto-pos-weight --amp
```

Useful flags:

```text
--pretrained          use ImageNet MobileNetV3-small encoder weights
--auto-pos-weight     estimate BCE positive-class weight from training masks
--hflip               enable horizontal flip augmentation
--freeze-encoder      train only the decoder/head at first
--resume PATH         resume from a checkpoint
```

Checkpoints and logs are saved in `checkpoints/`:

```text
best.pth
latest.pth
history.csv
config.json
```

## Infer One Image

```powershell
python src\infer_one.py --image data\images\frame_0001.jpg --ckpt checkpoints\best.pth --out-dir outputs
```

Outputs:

```text
*_prob.png
*_heat.png
*_mask.png
*_overlay.png
```

Run inference plus contour corner extraction:

```powershell
python src\infer_one.py --image data\images\frame_0001.jpg --ckpt checkpoints\best.pth --postprocess
```

## Postprocess Existing Mask

```powershell
python src\postprocess.py --mask outputs\frame_0001_mask.png --image data\images\frame_0001.jpg --out-dir outputs
```

Outputs:

```text
*_clean_mask.png
*_corners.png
*_corners.json
```

Corner order in JSON:

```text
top_left, top_right, bottom_right, bottom_left
```

## Notes

- Masks are binarized with `mask > 127`.
- Image resizing uses bilinear interpolation; mask resizing uses nearest-neighbor interpolation.
- Training loss is `BCEWithLogitsLoss + Dice loss`.
- The contour postprocess is intentionally simple. It is a first visualization step before LSD/Hough/RANSAC/PnP integration.
