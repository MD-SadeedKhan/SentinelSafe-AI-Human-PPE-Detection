#!/usr/bin/env python3
# Updated pascalVOC_to_yolo.py
import os
import argparse
import xml.etree.ElementTree as ET

def convert_box(size, box):
    w, h = size
    xmin, ymin, xmax, ymax = box
    x_center = (xmin + xmax) / 2.0 / w
    y_center = (ymin + ymax) / 2.0 / h
    box_w = (xmax - xmin) / w
    box_h = (ymax - ymin) / h
    return x_center, y_center, box_w, box_h

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="VOC annotations dir (XMLs)")
    parser.add_argument("--images_dir", required=True, help="Images dir (matching XML names)")
    parser.add_argument("--output_dir", required=True, help="Output YOLO labels dir")
    parser.add_argument("--classes_file", required=True, help="Path to classes.txt")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    with open(args.classes_file, "r") as f:
        classes = [c.strip() for c in f.readlines()]

    os.makedirs(args.output_dir, exist_ok=True)

    total_xml = 0
    total_labels = 0

    for xml_file in os.listdir(args.input_dir):
        if not xml_file.endswith(".xml"):
            continue

        total_xml += 1
        tree = ET.parse(os.path.join(args.input_dir, xml_file))
        root = tree.getroot()

        filename = root.find('filename').text
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        out_lines = []
        for obj in root.findall('object'):
            cls_name = obj.find('name').text.strip()
            if cls_name not in classes:
                continue
            cls_id = classes.index(cls_name)
            bndbox = obj.find('bndbox')
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            x_c, y_c, bw, bh = convert_box((w, h), (xmin, ymin, xmax, ymax))
            out_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}")

        # Create TXT file even if empty
        txt_name = os.path.splitext(filename)[0] + ".txt"
        with open(os.path.join(args.output_dir, txt_name), "w") as f:
            f.write("\n".join(out_lines))

        total_labels += 1

    print(f"Conversion completed! Total XML files processed: {total_xml}")
    print(f"Total TXT labels created: {total_labels}")
