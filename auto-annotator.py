import os
import argparse
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv
from supervision.draw.utils import calculate_optimal_line_thickness
from supervision.keypoint.skeletons import SKELETONS_BY_VERTEX_COUNT


# Initialize YOLOv8 pose estimator
MODEL_PATH = "yolov8x-pose.pt"
pose_model = YOLO(MODEL_PATH)

def process_image(image_path, labels_folder, annotated_folder):
    """ Process an image with YOLOv8 pose estimation and save annotations """
    img = Image.open(image_path)
    img_np = np.array(img)
    img_w, img_h = img.size

    results = pose_model(image_path)[0]

    if results.keypoints is None or results.keypoints.xy.numel() == 0:
        print(f"❌ No keypoints detected in {image_path}")
        return

    # Convert results to numpy arrays
    keypoints_np = results.keypoints.xy.cpu().numpy()
    bounding_boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []

    # Compute optimal thickness dynamically
    optimal_thickness = calculate_optimal_line_thickness(img_np.shape[:2])  # Uses image height & width
    optimal_radius = int(optimal_thickness * 1.5)  # Slightly scale the radius

    # Prepare keypoints for supervision
    keypoints_list = []
    for person in keypoints_np:
        person_keypoints = [(kp[0], kp[1]) for kp in person]  # Ensure (x, y) format
        keypoints_list.append(person_keypoints)

    keypoints_array = np.array(keypoints_list)  # Convert list to NumPy array

    # Ensure correct shape for supervision
    keypoints_sv = sv.KeyPoints(
        xy=keypoints_array,
        confidence=np.ones_like(keypoints_array[:, :, 0])  # Create dummy confidence values
    )

    # Automatically determine the correct skeleton structure
    skeleton_edges = SKELETONS_BY_VERTEX_COUNT.get(len(keypoints_sv.xy[0]), None)

    # Create annotators using **dynamic thickness & radius**
    box_annotator = sv.BoxAnnotator(thickness=optimal_thickness)
    vertex_annotator = sv.VertexAnnotator(radius=optimal_radius, color=sv.Color.RED)
    edge_annotator = sv.EdgeAnnotator(thickness=optimal_thickness, edges=skeleton_edges, color=sv.Color.BLUE)

    # Prepare bounding box detections for supervision
    detections = sv.Detections(
                xyxy=results.boxes.xyxy.cpu().numpy(),
                class_id=np.arange(len(results.boxes.xyxy))  # Unique ID per person
            )

    # Apply keypoint and skeleton visualization
    img_np = box_annotator.annotate(scene=img_np, detections=detections)
    img_np = vertex_annotator.annotate(scene=img_np, key_points=keypoints_sv)
    img_np = edge_annotator.annotate(scene=img_np, key_points=keypoints_sv)

    # Save annotated image
    annotated_img_path = os.path.join(annotated_folder, os.path.basename(image_path).replace(".jpg", "_annotated.png"))
    Image.fromarray(img_np).save(annotated_img_path)

    # Save keypoints in YOLO format
    annotation_file = os.path.join(labels_folder, os.path.basename(image_path).replace(".jpg", ".txt"))
    
    with open(annotation_file, "w") as f:
        for person_idx, person_keypoints in enumerate(keypoints_np):
            x_values = [kp[0] for kp in person_keypoints if kp[0] > 0]
            y_values = [kp[1] for kp in person_keypoints if kp[1] > 0]

            if not x_values or not y_values:
                continue

            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)

            bbox_x = (x_min + x_max) / 2 / img_w
            bbox_y = (y_min + y_max) / 2 / img_h
            bbox_w = (x_max - x_min) / img_w
            bbox_h = (y_max - y_min) / img_h

            keypoints_normalized = []
            for x, y in person_keypoints:
                if x > 0 and y > 0:
                    keypoints_normalized.extend([x / img_w, y / img_h, 2])
                else:
                    keypoints_normalized.extend([0.0, 0.0, 0.0])

            f.write(f"0 {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f} " + " ".join(f"{kp:.6f}" for kp in keypoints_normalized) + "\n")

    print(f"✅ Processed: {image_path}")

def process_folder(image_folder):
    """ Process all images in a folder """
    labels_folder = os.path.join(os.path.dirname(image_folder), "labels")
    annotated_folder = os.path.join(os.path.dirname(image_folder), "annotated_images")
    
    os.makedirs(labels_folder, exist_ok=True)
    os.makedirs(annotated_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]

    if not image_files:
        print(f"⚠️ No images found in {image_folder}")
        return

    for image_file in tqdm(image_files, desc=f"Processing {image_folder}"):
        process_image(os.path.join(image_folder, image_file), labels_folder, annotated_folder)

def main():
    parser = argparse.ArgumentParser(description="Batch pose annotation tool using YOLOv8x-pose.")
    parser.add_argument("folders", nargs="+", help="Paths to image folders.")
    args = parser.parse_args()

    for folder in args.folders:
        if os.path.exists(folder) and os.path.isdir(folder):
            process_folder(folder)
        else:
            print(f"❌ Invalid folder: {folder}")

if __name__ == "__main__":
    main()
