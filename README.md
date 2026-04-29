# VIO
Small U-Net / MobileNetV3-UNet / Fast-SCNN
384×384 RGB image  crop  ->  384×384 gate mask
loss：BCE + Dice Loss
post：mask → threshold → morphology → LSD/Hough/contour → corners
