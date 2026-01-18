import cv2, os

def preprocess_folder(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for f in os.listdir(src_folder):
        if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        img = cv2.imread(os.path.join(src_folder, f))
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l,a,b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        enhanced = cv2.merge((cl,a,b))
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        cv2.imwrite(os.path.join(dst_folder, f), enhanced)
    print(f"Processed images saved to {dst_folder}")

preprocess_folder("images/train", "images/train_processed")
preprocess_folder("images/val", "images/val_processed")
