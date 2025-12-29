# 🚨 **SentinelSafe AI – Human & PPE Detection System**  
### 🧠 Real-Time Human Detection + Safety Equipment Compliance Analysis  
*A Deep Learning–powered safety intelligence system built using YOLOv8.*

---

## 📌 **Project Overview**

**SentinelSafe AI** is a computer-vision system designed to enhance workplace safety by:

- 👷 Detecting **humans (person detection)**  
- 🦺 Identifying **PPE equipment** such as hard-hat, gloves, mask, glasses, boots, vest, PPE-suit, ear-protector, safety-harness  
- 🖼️ Running inference on real images and manually drawing bounding boxes  
- 🔍 Providing high-accuracy predictions using **two separately trained YOLOv8 models**  

This system ensures safety compliance in industrial & high-risk zones, enabling smarter monitoring and automated reporting.

---

## ⭐ **Key Features**

✔️ Person Detection Model (YOLOv8)  
✔️ PPE Detection Model on **cropped person images**  
✔️ Clean and manual bounding-box drawing using OpenCV  
✔️ Annotation converter (PascalVOC → YOLO)  
✔️ Fully modular inference pipeline  
✔️ Industry-grade evaluation metrics (Precision, Recall, mAP50, mAP50-95)  
✔️ Professional project report included  

---

## 🛠️ **Tech Stack**

| Technology | Usage |
|-----------|--------|
| 🐍 **Python** | Core development |
| 🔍 **YOLOv8 (Ultralytics)** | Object detection models |
| 📦 **Conda** | Environment management |
| 🖼️ **OpenCV** | Manual bounding-box drawing |
| 📝 **Pascal VOC** | Original dataset format |
| 🔧 **argparse** | CLI arguments for scripts |

---

## 📁 **Folder Structure**

```
SentinelSafe-AI/
│
├── pascalVOC_to_yolo.py         # Convert Pascal VOC annotations to YOLO format
├── inference.py                 # Manual bounding box inference pipeline
├── yolov8_ppe.yaml              # PPE detection dataset config
├── yolov8_data_person.yaml      # Person detection dataset config
├── create_yolo_labels.py        # Utility script
├── crop_persons.py              # Crop persons for PPE model training
│
├── requirements.txt             # Python dependencies
│
├── Report.pdf                   # Detailed project & analysis report
```

🚫 **Note:**  
Weights folder (`weights/`) is intentionally not included due to size.  
Users can train new weights following the instructions below.

---

## 🎯 **Project Workflow**

### **1️⃣ Annotation Conversion**
Converts VOC XML annotations into YOLOv8 `.txt` format.

```
python pascalVOC_to_yolo.py --input_dir path/to/xmls \
                            --images_dir path/to/images \
                            --output_dir labels_yolo \
                            --classes_file classes.txt
```

---

### **2️⃣ Model Training**

#### ✔️ Train Person Model
```
yolo detect train model=yolov8n.pt data=yolov8_data_person.yaml epochs=50 imgsz=640 batch=16 name=person_train
```

#### ✔️ Train PPE Model
```
yolo detect train model=yolov8n.pt data=yolov8_ppe.yaml epochs=50 imgsz=640 batch=16 name=ppe_train
```

---

### **3️⃣ Inference Pipeline**
Run detection using both models:

```
python inference.py \
  --input_dir sample_images \
  --output_dir results \
  --person_det_model weights/person.pt \
  --ppe_det_model weights/ppe.pt
```

---

## 📊 **Evaluation Metrics**

### **🟢 Person Model Performance**
| Metric | Score |
|--------|--------|
| Precision | **0.981** |
| Recall | **0.955** |
| mAP50 | **0.989** |
| mAP50-95 | **0.896** |

---

### **🟣 PPE Model Performance**
| Metric | Score |
|--------|--------|
| Precision | **1.0** |
| Recall | **1.0** |
| mAP50 | **0.995** |
| mAP50-95 | **0.995** |

📌 *Graphs and visualization results are included inside* **Report.pdf**.

---

## 📥 Installation

### **1️⃣ Create Conda Environment**
```
conda create -n sentinelsafe python=3.10 -y
conda activate sentinelsafe
```

### **2️⃣ Install Requirements**
```
pip install -r requirements.txt
```

---

## 🧪 Sample Output  

🚀 Bounding boxes for both human and PPE are drawn **manually** using OpenCV.  
Add your own results images below if needed.

---

## 🤝 Contributing

Pull requests are welcome!  
If you'd like to improve detection accuracy or add new safety classes, feel free to contribute.

---

## 📄 License
This project is released under the **MIT License**.

---

## 👤 Author
**Sadeed Khan**  
📌 *Data Science & AI Student*  
📌 *Focused on Computer Vision, AI Systems, and Deep Learning Projects*
