import cv2
import numpy as np
import os

# Folders
img_folder = "images/train_processed"
label_folder = "labels/train"
os.makedirs(label_folder, exist_ok=True)

# YOLO image size (for normalization)
IMG_W, IMG_H = 640, 640

# Loop through images
for img_name in os.listdir(img_folder):
    if not img_name.lower().endswith(('.png','.jpg','.jpeg')):
        continue
    
    img_path = os.path.join(img_folder, img_name)
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (IMG_W, IMG_H))

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Threshold to detect bright microplastics
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    
    # Find contours/blobs
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    yolo_labels = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 2 or h < 2:  # ignore tiny noise
            continue
        
        # Normalize to 0-1
        x_center = (x + w/2) / IMG_W
        y_center = (y + h/2) / IMG_H
        width = w / IMG_W
        height = h / IMG_H
        
        yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Save YOLO label file
    label_path = os.path.join(label_folder, os.path.splitext(img_name)[0]+".txt")
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_labels))
    
    print(f"Labeled {img_name}: {len(yolo_labels)} particles")
