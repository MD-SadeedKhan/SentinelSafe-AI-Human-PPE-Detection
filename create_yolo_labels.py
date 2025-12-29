import os
import shutil
import random

# ----------------- CONFIG -----------------
images_dir = "datasets/images"           # Cropped images folder
labels_dir = "datasets/labels"           # Output labels folder
classes_file = "datasets/classes.txt"    # File with all classes
train_ratio = 0.8                         # 80% train, 20% val
random_seed = 42                         

# ----------------- SETUP -----------------
os.makedirs(labels_dir, exist_ok=True)

# Read classes
with open(classes_file, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# ----------------- CREATE LABELS -----------------
count = 0
for img_file in os.listdir(images_dir):
    if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    txt_file = os.path.splitext(img_file)[0] + ".txt"
    txt_path = os.path.join(labels_dir, txt_file)

    # For cropped images, assume class 0 (person) fills the image
    with open(txt_path, "w") as f:
        f.write("0 0.5 0.5 1.0 1.0\n")

    count += 1

print(f"Labels created for {count} cropped images")

# ----------------- SPLIT TRAIN / VAL -----------------
all_images = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
random.seed(random_seed)
random.shuffle(all_images)

split_idx = int(len(all_images) * train_ratio)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Create train/val folders
for folder in ["train", "val"]:
    os.makedirs(os.path.join(images_dir, folder), exist_ok=True)
    os.makedirs(os.path.join(labels_dir, folder), exist_ok=True)

# Move images and labels
for img_file in train_images:
    shutil.move(os.path.join(images_dir, img_file), os.path.join(images_dir, "train", img_file))
    shutil.move(os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt"),
                os.path.join(labels_dir, "train", os.path.splitext(img_file)[0] + ".txt"))

for img_file in val_images:
    shutil.move(os.path.join(images_dir, img_file), os.path.join(images_dir, "val", img_file))
    shutil.move(os.path.join(labels_dir, os.path.splitext(img_file)[0] + ".txt"),
                os.path.join(labels_dir, "val", os.path.splitext(img_file)[0] + ".txt"))

print(f"\nDataset split completed!")
print(f"Train images: {len(train_images)}, Val images: {len(val_images)}")
