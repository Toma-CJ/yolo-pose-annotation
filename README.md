# YOLO Pose Annotation Toolset

## Overview
This repository provides a toolset for **automatic** and **semi-automatic** pose annotation to fine-tune **YOLO pose models** with custom datasets. The tool allows users to generate annotations efficiently and refine them as needed before training a fine-tuned YOLO pose model.

### Features
- **Fully automatic annotation**: Uses `YOLOv8x-pose` to detect keypoints and generate annotations automatically.
- **Semi-automatic annotation**: Provides an interactive tool to review and correct annotations.
- **Fine-tuning pipeline**: Facilitates the fine-tuning of YOLO pose models with custom datasets.

## Getting started

### 1. Clone the Repository
```bash
git clone https://github.com/kevgildea/yolo-pose-annotation.git
cd yolo-pose-annotation
```

### 2. Install Dependencies
Make sure you have Python installed (\>=3.8). Then, install the required packages:
```bash
pip install -r requirements.txt
```

### 3. Prepare Your Dataset
Organize your dataset in the following structure:
```
📂 your-dataset/
 ├── 📂 train/
 │   ├── images/
 │   │   ├── image1.jpg
 │   │   ├── image2.jpg
 │   │   └── ...
 │   └── labels/
 │       ├── image1.txt
 │       ├── image2.txt
 │       └── ...
 ├── 📂 val/
 │   ├── images/
 │   │   ├── image1.jpg
 │   │   ├── image2.jpg
 │   │   └── ...
 │   └── labels/
 │       ├── image1.txt
 │       ├── image2.txt
 │       └── ...
 ├── 📂 test/
 │   ├── images/
 │   │   ├── image1.jpg
 │   │   ├── image2.jpg
 │   │   └── ...
 │   └── labels/
 │       ├── image1.txt
 │       ├── image2.txt
 │       └── ...
```


**UPDATE:**


Each `.txt` file should follow the YOLO pose annotation format (normalised coordinates):
```
<class_id> <bbox_x> <bbox_y> <bbox_w> <bbox_h> <kp1_x> <kp1_y> <visibility> ...
```

### 4. Run Automatic Annotation
To generate automatic annotations, run:
```bash
python automatic_annotator.py --image_folder path/to/images --output_folder path/to/labels
```

### 5. Run Semi-Automatic Annotation
To review and refine annotations interactively:
```bash
python semi_auto_annotator.py --image_folder path/to/images --label_folder path/to/labels
```

### 6. Fine-Tune YOLO Pose Model
To train a YOLO pose model with the annotated dataset:
```bash
python train.py --data custom_data.yaml --epochs 50 --weights yolov8s-pose.pt
```


## Contributing
Feel free to submit pull requests or open issues for improvements and bug fixes!
