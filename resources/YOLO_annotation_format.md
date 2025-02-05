# YOLOv8-Pose annotation format

Each image has a corresponding `.txt` file in with the following format:

```
<class_id> <bbox_x_center> <bbox_y_center> <bbox_width> <bbox_height> <x1> <y1> <v1> <x2> <y2> <v2> ... <xN> <yN> <vN>
```

Where:
- `<class_id>` → Object class index (typically `0` for a single-class pose dataset)
- `<bbox_x_center> <bbox_y_center> <bbox_width> <bbox_height>` → Normalised bounding box values (between `0` and `1`)
- `<xN> <yN> <vN>` → Normalised keypoints (x, y, visibility)

**Example annotation for a single person with 17 keypoints (COCO format):**

```
0 0.5 0.5 0.4 0.6  0.52 0.40 2  0.54 0.38 2  0.55 0.36 1  0.50 0.70 2  0.48 0.88 2  0.45 0.95 2 ...
```

- `0` → Class ID
- `0.5 0.5 0.4 0.6` → Bounding box center `(x, y)`, width, height
- `x1, y1, v1, x2, y2, v2, ...` → Normalised keypoints

If an image has **multiple people**, each person is written on a new line.


The standard COCO 17-keypoint order used in YOLOv8-Pose is:

1. Nose
2. Left Eye
3. Right Eye
4. Left Ear
5. Right Ear
6. Left Shoulder
7. Right Shoulder
8. Left Elbow
9. Right Elbow
10. Left Wrist
11. Right Wrist
12. Left Hip
13. Right Hip
14. Left Knee
15. Right Knee
16. Left Ankle
17. Right Ankle

Each keypoint has three values: `<x> <y> <visibility>`.
- `x` and `y` are normalised between `0` and `1`.
- `visibility` is:
  - `0` → Not visible
  - `1` → Occluded
  - `2` → Fully visible


