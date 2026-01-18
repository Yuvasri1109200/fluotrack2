from ultralytics.yolo.utils.augmentations import Albumentations
import cv2, os

alb = Albumentations(
    transforms=[
        dict(type='HueSaturationValue', hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        dict(type='RandomScale', scale_limit=0.2, p=0.5),
        dict(type='GaussNoise', var_limit=(10.0,50.0), p=0.3)
    ]
)

def augment_folder(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for f in os.listdir(src_folder):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(os.path.join(src_folder, f))
        augmented = alb(img=img)['image']
        cv2.imwrite(os.path.join(dst_folder, f), augmented)
    print(f"Augmented images saved to {dst_folder}")
