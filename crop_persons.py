#!/usr/bin/env python3
import os
import cv2
from ultralytics import YOLO

# ------------------- MODIFY THESE PATHS -------------------
MODEL_PATH = r"person_det/train3/weights/best.pt"
IMAGE_DIR  = r"datasets/images"
OUTPUT_DIR = r"datasets/cropped_persons"
CONF_THRESH = 0.30  # confidence threshold
# ----------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

for img_name in os.listdir(IMAGE_DIR):
    if not img_name.lower().endswith((".jpg",".png",".jpeg")):
        continue
    
    img_path = os.path.join(IMAGE_DIR, img_name)
    results = model.predict(img_path, imgsz=640, conf=CONF_THRESH)

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    count = 1
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf = float(box.conf)

            if cls == 0 and conf >= CONF_THRESH: 
                x1,y1,x2,y2 = map(int, box.xyxy[0])

                crop = img[y1:y2, x1:x2]
                if crop.size == 0: continue

                save_name = f"{img_name.split('.')[0]}_person_{count}.jpg"
                cv2.imwrite(os.path.join(OUTPUT_DIR, save_name), crop)
                print(f"SAVED → {save_name}")
                count += 1

print("\n==== CROPPING FINISHED ====")
print(f"Saved in: {OUTPUT_DIR}")
