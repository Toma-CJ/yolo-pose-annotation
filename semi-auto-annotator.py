import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from ultralytics import YOLO
import os
import json
import numpy as np
import supervision as sv
from supervision.keypoint.skeletons import SKELETONS_BY_VERTEX_COUNT
from supervision.draw.utils import calculate_optimal_line_thickness

class ImageAnnotator:
    def __init__(self, root):
        self.root = root
        self.root.title("Semi-automatic Pose Annotator (YOLO-format)")

        # Initialize YOLOv8 pose estimator
        self.pose_estimator = YOLO('yolov8x-pose.pt')

        # Frame to hold canvas and scrollbars
        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=tk.YES)

        # Scrollbars
        self.x_scroll = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL)
        self.x_scroll.grid(row=1, column=0, sticky=tk.EW)
        self.y_scroll = tk.Scrollbar(self.frame, orient=tk.VERTICAL)
        self.y_scroll.grid(row=0, column=1, sticky=tk.NS)

        # Canvas for displaying the image
        self.canvas = tk.Canvas(self.frame, xscrollcommand=self.x_scroll.set, yscrollcommand=self.y_scroll.set)
        self.canvas.grid(row=0, column=0, sticky=tk.NSEW)

        # Configure frame to expand canvas
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        self.x_scroll.config(command=self.canvas.xview)
        self.y_scroll.config(command=self.canvas.yview)

        # Bind canvas for panning
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.pan_image)

        # Bind canvas for editing keypoints
        self.canvas.bind("<Button-1>", self.check_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        # Data storage
        self.image_paths = []  # List of image paths
        self.current_index = -1  # Current image index
        self.auto_keypoints = []  # Stores keypoints for all detected people
        self.selected_keypoint = None
        self.image = None  # Store loaded image
        self.tk_image = None  # Store Tkinter-compatible image

        self.keypoints_order = [
            "nose", "left_eye", "right_eye", "left_ear", "right_ear",
            "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip",
            "left_knee", "right_knee", "left_ankle", "right_ankle"
        ]

        # Menu options
        menu = tk.Menu(root)
        root.config(menu=menu)

        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Folder...", command=self.load_folder)
        file_menu.add_command(label="Save Annotations...", command=self.save_annotations)
        file_menu.add_command(label="Save Annotated Image", command=self.save_annotated_image)
        file_menu.add_command(label="Exit", command=root.quit)

        # Navigation buttons
        nav_frame = tk.Frame(root)
        nav_frame.pack()

        self.prev_button = tk.Button(nav_frame, text="Previous", command=self.prev_image, state=tk.DISABLED)
        self.prev_button.pack(side=tk.LEFT, padx=5)

        self.next_button = tk.Button(nav_frame, text="Next", command=self.next_image, state=tk.DISABLED)
        self.next_button.pack(side=tk.LEFT, padx=5)

        self.selected_bbox = None
        self.bbox_mode = None  # "move" or "resize"
        self.bbox_resize_corner = None  # Stores which corner is being resized
        self.resize_margin = 10  # Margin to detect corner click


    def load_folder(self):
        """ Selects a folder of images and loads them one by one """
        folder_selected = filedialog.askdirectory()
        if not folder_selected:
            return

        self.image_paths = [os.path.join(folder_selected, f) for f in os.listdir(folder_selected) if f.endswith(('.jpg', '.png', '.jpeg'))]
        self.image_paths.sort()

        if not self.image_paths:
            messagebox.showerror("Error", "No images found in the folder.")
            return

        self.current_index = 0
        self.load_image()

        # Enable buttons if more than one image
        if len(self.image_paths) > 1:
            self.next_button.config(state=tk.NORMAL)


    def load_image(self):
        """ Loads the current image and runs pose estimation """
        if self.current_index < 0 or self.current_index >= len(self.image_paths):
            return

        self.file_path = self.image_paths[self.current_index]
        self.canvas.delete("all")
        self.image = Image.open(self.file_path)
        image_np = np.array(self.image)

        # Run YOLO pose detection
        results = self.pose_estimator(self.file_path)[0]

        if results.boxes and results.keypoints.xy.numel() > 0:
            
            # Convert back to PIL for Tkinter
            self.image = Image.fromarray(image_np)
            self.tk_image = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

            # Process keypoints for each detected person
            self.auto_keypoints = []
            for person_idx, keypoints in enumerate(results.keypoints.xy):
                person_keypoints = []
                for xy, name in zip(keypoints, self.keypoints_order):
                    x, y = int(xy[0].item()), int(xy[1].item())
                    oval = self.canvas.create_oval(x-2, y-2, x+2, y+2, fill='green')
                    label = self.canvas.create_text(x+15, y, text=f"{name} (P{person_idx})", anchor='w', font=("Arial", 8), fill='red')
                    person_keypoints.append((x, y, oval, label))
                self.auto_keypoints.append(person_keypoints)
            self.auto_bboxes = []  # Stores bounding boxes for detected people
            for person_idx, (xyxy, keypoints) in enumerate(zip(results.boxes.xyxy.cpu().numpy(), results.keypoints.xy)):
                x_min, y_min, x_max, y_max = map(int, xyxy)  # Convert to int
                bbox_id = self.canvas.create_rectangle(x_min, y_min, x_max, y_max, outline="blue", width=2)
                self.auto_bboxes.append((x_min, y_min, x_max, y_max, bbox_id))


        # Update navigation button states
        self.prev_button.config(state=tk.NORMAL if self.current_index > 0 else tk.DISABLED)
        self.next_button.config(state=tk.NORMAL if self.current_index < len(self.image_paths) - 1 else tk.DISABLED)

        self.save_annotations(auto=True)


    def save_annotations(self, auto=False):
        """ Saves keypoints in YOLO format in a 'labels' folder for each image """
        
        labels_folder = os.path.join(os.path.dirname(os.path.dirname(self.file_path)), "labels")
        os.makedirs(labels_folder, exist_ok=True)

        annotation_filename = os.path.join(
            labels_folder,
            os.path.basename(self.file_path).replace(".jpg", ".txt").replace(".png", ".txt")
        )

        img_w, img_h = self.image.size  # Image dimensions

        with open(annotation_filename, "w") as f:
            for person_idx, person_keypoints in enumerate(self.auto_keypoints):
                # Get bounding box from keypoints
                x_values = [kp[0] for kp in person_keypoints if kp[0] > 0]
                y_values = [kp[1] for kp in person_keypoints if kp[1] > 0]

                if not x_values or not y_values:
                    continue  # Skip if no valid keypoints

                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)

                bbox_x = (x_min + x_max) / 2  # Center X
                bbox_y = (y_min + y_max) / 2  # Center Y
                bbox_w = x_max - x_min  # Width
                bbox_h = y_max - y_min  # Height

                # Normalize bounding box
                bbox_x /= img_w
                bbox_y /= img_h
                bbox_w /= img_w
                bbox_h /= img_h

                # Format keypoints (normalize and add visibility)
                keypoints = []
                for x, y, _, _ in person_keypoints:
                    if x > 0 and y > 0:  # If keypoint is valid
                        x_norm = x / img_w
                        y_norm = y / img_h
                        visibility = 2.0
                    else:  # Missing keypoints
                        x_norm, y_norm, visibility = 0.000000, 0.000000, 0.000000

                    keypoints.extend([x_norm, y_norm, visibility])

                # Write annotation in YOLO format
                f.write(f"{person_idx} {bbox_x:.6f} {bbox_y:.6f} {bbox_w:.6f} {bbox_h:.6f} " +
                        " ".join(f"{kp:.6f}" for kp in keypoints) + "\n")

        print(f"Annotations saved: {annotation_filename}")

    def next_image(self):
        """ Moves to the next image in the folder """
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        """ Moves to the previous image in the folder """
        if self.current_index > 0:
            self.current_index -= 1
            self.load_image()

    def start_pan(self, event):
        """ Starts panning the image """
        self.canvas.scan_mark(event.x, event.y)

    def pan_image(self, event):
        """ Handles panning movement """
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def check_click(self, event):
        """Detects if a keypoint or bounding box was clicked for dragging or resizing."""
        self.selected_keypoint = None
        self.selected_bbox = None
        self.bbox_mode = None
        self.bbox_resize_corner = None
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Check bounding boxes
        for i, (x_min, y_min, x_max, y_max, bbox_id) in enumerate(self.auto_bboxes):
            width = x_max - x_min
            height = y_max - y_min

            # Define edge thickness for selection
            edge_margin = max(5, int(min(width, height) * 0.05))  # Dynamically adjust

            # Detect corner clicks for resizing
            if abs(canvas_x - x_min) < edge_margin and abs(canvas_y - y_min) < edge_margin:
                self.selected_bbox = (self.auto_bboxes, i, bbox_id)
                self.bbox_mode = "resize"
                self.bbox_resize_corner = "top_left"
                return
            elif abs(canvas_x - x_max) < edge_margin and abs(canvas_y - y_min) < edge_margin:
                self.selected_bbox = (self.auto_bboxes, i, bbox_id)
                self.bbox_mode = "resize"
                self.bbox_resize_corner = "top_right"
                return
            elif abs(canvas_x - x_min) < edge_margin and abs(canvas_y - y_max) < edge_margin:
                self.selected_bbox = (self.auto_bboxes, i, bbox_id)
                self.bbox_mode = "resize"
                self.bbox_resize_corner = "bottom_left"
                return
            elif abs(canvas_x - x_max) < edge_margin and abs(canvas_y - y_max) < edge_margin:
                self.selected_bbox = (self.auto_bboxes, i, bbox_id)
                self.bbox_mode = "resize"
                self.bbox_resize_corner = "bottom_right"
                return

            # Detect clicks on edges (for moving)
            elif (
                (abs(canvas_x - x_min) < edge_margin or abs(canvas_x - x_max) < edge_margin) or
                (abs(canvas_y - y_min) < edge_margin or abs(canvas_y - y_max) < edge_margin)
            ):
                self.selected_bbox = (self.auto_bboxes, i, bbox_id, (canvas_x, canvas_y))
                self.bbox_mode = "move"
                return

        # Check if clicking a keypoint (if no bounding box is selected)
        for person_keypoints in self.auto_keypoints:
            for i, (x, y, oval, label) in enumerate(person_keypoints):
                if abs(canvas_x - x) < 10 and abs(canvas_y - y) < 10:
                    self.selected_keypoint = (person_keypoints, i, oval, label)
                    return

    def on_drag(self, event):
        """Moves or resizes a bounding box or moves a keypoint while dragging."""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)

        # Resize bounding box
        if self.selected_bbox and self.bbox_mode == "resize":
            bbox_list, index, bbox_id = self.selected_bbox
            x_min, y_min, x_max, y_max, _ = bbox_list[index]

            # Adjust the correct corner
            if self.bbox_resize_corner == "top_left":
                x_min, y_min = canvas_x, canvas_y
            elif self.bbox_resize_corner == "top_right":
                x_max, y_min = canvas_x, canvas_y
            elif self.bbox_resize_corner == "bottom_left":
                x_min, y_max = canvas_x, canvas_y
            elif self.bbox_resize_corner == "bottom_right":
                x_max, y_max = canvas_x, canvas_y

            # Update bounding box on canvas
            self.canvas.coords(bbox_id, x_min, y_min, x_max, y_max)

            # Update stored coordinates
            bbox_list[index] = (x_min, y_min, x_max, y_max, bbox_id)

        # Move bounding box (only if clicked on edge)
        elif self.selected_bbox and self.bbox_mode == "move":
            bbox_list, index, bbox_id, (prev_x, prev_y) = self.selected_bbox
            dx = canvas_x - prev_x
            dy = canvas_y - prev_y
            x_min, y_min, x_max, y_max, _ = bbox_list[index]

            # Move the box
            new_x_min = x_min + dx
            new_y_min = y_min + dy
            new_x_max = x_max + dx
            new_y_max = y_max + dy

            # Update bounding box position on canvas
            self.canvas.coords(bbox_id, new_x_min, new_y_min, new_x_max, new_y_max)

            # Update stored coordinates
            bbox_list[index] = (new_x_min, new_y_min, new_x_max, new_y_max, bbox_id)
            self.selected_bbox = (bbox_list, index, bbox_id, (canvas_x, canvas_y))

        # Move keypoint (this part remains the same)
        elif self.selected_keypoint:
            person_keypoints, index, oval, label = self.selected_keypoint
            self.canvas.coords(oval, canvas_x-2, canvas_y-2, canvas_x+2, canvas_y+2)
            self.canvas.coords(label, canvas_x + 15, canvas_y)
            person_keypoints[index] = (canvas_x, canvas_y, oval, label)

    def on_release(self, event):
        """Stops dragging and saves new bounding box coordinates."""
        self.selected_keypoint = None
        self.selected_bbox = None
        self.bbox_mode = None
        self.bbox_resize_corner = None

    def save_annotated_image(self):
        """Saves the annotated image with optimized keypoints & skeleton visualization using Supervision."""
        if self.image is None:
            messagebox.showerror("Error", "No image loaded.")
            return

        # Convert PIL image to numpy array
        image_np = np.array(self.image)

        # Compute optimal thickness dynamically
        optimal_thickness = calculate_optimal_line_thickness(image_np.shape[:2])  # Uses image height & width
        optimal_radius = int(optimal_thickness * 1.5)  # Slightly scale the radius

        # Convert manually adjusted bounding boxes to a numpy array
        adjusted_bboxes = np.array([
            [x_min, y_min, x_max, y_max] for x_min, y_min, x_max, y_max, _ in self.auto_bboxes
        ])

        # Create Supervision detections object for updated bounding boxes
        detections = sv.Detections(
            xyxy=adjusted_bboxes,
            class_id=np.arange(len(adjusted_bboxes))  # Assign unique IDs to each box
        )

        # Draw bounding boxes using Supervision
        annotator = sv.BoxAnnotator()
        image_np = annotator.annotate(scene=image_np, detections=detections)

        # Extract keypoints properly as a NumPy array
        keypoints_list = []
        for person in self.auto_keypoints:
            person_keypoints = [(kp[0], kp[1]) for kp in person]  # Ensure (x, y) format
            keypoints_list.append(person_keypoints)

        keypoints_array = np.array(keypoints_list)  # Convert list to NumPy array

        # Ensure correct shape for supervision
        keypoints = sv.KeyPoints(
            xy=keypoints_array,
            confidence=np.ones_like(keypoints_array[:, :, 0])  # Create dummy confidence values
        )

        # Automatically determine the correct skeleton structure
        skeleton_edges = SKELETONS_BY_VERTEX_COUNT.get(len(keypoints.xy[0]), None)

        # Create annotators using **dynamic thickness & radius**
        vertex_annotator = sv.VertexAnnotator(radius=optimal_radius, color=sv.Color.RED)
        edge_annotator = sv.EdgeAnnotator(thickness=optimal_thickness, edges=skeleton_edges, color=sv.Color.BLUE)

        # Apply keypoint and skeleton visualization
        image_np = edge_annotator.annotate(scene=image_np, key_points=keypoints)
        image_np = vertex_annotator.annotate(scene=image_np, key_points=keypoints)

        # Convert back to PIL image
        annotated_image = Image.fromarray(image_np)

        # Save the annotated image
        annotated_folder = os.path.join(os.path.dirname(os.path.dirname(self.file_path)), "annotated_images")
        os.makedirs(annotated_folder, exist_ok=True)
        annotated_image_path = os.path.join(annotated_folder, os.path.basename(self.file_path).replace(".jpg", "_annotated.png"))

        annotated_image.save(annotated_image_path)
        messagebox.showinfo("Success", "Annotated image saved successfully!")


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAnnotator(root)
    root.mainloop()


