from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train/weights/best.pt")
img = cv2.imread("images/test/sample.jpg")
results = model(img)[0]

for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,0,255), 2)

cv2.imshow("Detections", img)
cv2.waitKey(0)
