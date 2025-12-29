import os
import cv2
import argparse
from ultralytics import YOLO

# Define class names
person_classes = ["person"]
ppe_classes = ["hard-hat","gloves","mask","glasses","boots","vest","ppe-suit","ear-protector","safety-harness"]

def draw_boxes(image, boxes, classes, color=(0,255,0), label_prefix=""):
    for box, cls_id, conf in boxes:
        x1, y1, x2, y2 = map(int, box)
        label = f"{label_prefix}{classes[cls_id]} {conf:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image

def extract_boxes(results):
    """Extract boxes as list of tuples: (xyxy, class_id, confidence)"""
    boxes_list = []
    for det in results:
        if len(det.boxes) == 0:
            continue
        for i in range(len(det.boxes)):
            xyxy = det.boxes.xyxy[i].cpu().numpy()
            cls_id = int(det.boxes.cls[i].cpu().numpy())
            conf = float(det.boxes.conf[i].cpu().numpy())
            boxes_list.append((xyxy, cls_id, conf))
    return boxes_list

def main(input_dir, output_dir, person_model_path, ppe_model_path):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading models...")
    person_model = YOLO(person_model_path)
    ppe_model = YOLO(ppe_model_path)

    for img_name in os.listdir(input_dir):
        if not img_name.lower().endswith((".jpg",".png",".jpeg")):
            continue

        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        # PERSON DETECTION
        person_results = person_model(image)
        person_boxes = extract_boxes(person_results)

        for xyxy, cls_id, conf in person_boxes:
            x1, y1, x2, y2 = map(int, xyxy)
            # Crop person for PPE detection
            crop = image[y1:y2, x1:x2]
            ppe_results = ppe_model(crop)
            ppe_boxes = extract_boxes(ppe_results)

            # Draw PPE boxes on original image with coordinates adjusted
            for p_xyxy, p_cls_id, p_conf in ppe_boxes:
                px1, py1, px2, py2 = map(int, p_xyxy)
                # Adjust coordinates relative to original image
                cv2.rectangle(image, (x1+px1, y1+py1), (x1+px2, y1+py2), (0,0,255), 2)
                label = f"PPE_{ppe_classes[p_cls_id]} {p_conf:.2f}"
                cv2.putText(image, label, (x1+px1, y1+py1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        # Draw person boxes
        draw_boxes(image, person_boxes, person_classes, color=(0,255,0), label_prefix="Person_")

        # Save output
        cv2.imwrite(os.path.join(output_dir, img_name), image)

    print(f"Saved annotated images in: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--person_det_model", required=True)
    parser.add_argument("--ppe_det_model", required=True)
    args = parser.parse_args()

    main(args.input_dir, args.output_dir, args.person_det_model, args.ppe_det_model)
