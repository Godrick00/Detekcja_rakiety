from ultralytics import YOLO
import csv
from pathlib import Path

# -----------------------------
# Step 1: Load video file
# -----------------------------
video_path = Path("input_video.mp4")

if not video_path.exists():
    raise FileNotFoundError(f"Video file not found: {video_path}")
print(f"Loaded file: {video_path}")


# -----------------------------
# Step 2: Load YOLO models
# -----------------------------
print("Loading YOLOv8 Pose model...")
pose_model = YOLO("yolov8l-pose.pt")

print("Loading YOLOv8 World model...")
racket_model = YOLO("yolov8l-world.pt")
racket_model.set_classes(["sports racket"])


# -----------------------------
# Step 3: Create output directory
# -----------------------------
output_dir = Path("yolo_results")
output_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Step 4: Detect poses and keypoints
# -----------------------------
print("Running pose detection...")
pose_results = pose_model.track(
    source=str(video_path),
    save=False,
    verbose=True,
    classes=[0],  # Person class
    conf=0.5
)

# -----------------------------
# Step 5: Save keypoints to CSV
# -----------------------------
csv_path = output_dir / "keypoints_data.csv"
print(f"Saving keypoints to: {csv_path}")

keypoint_names = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle"
]

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "track_id", "keypoint_id", "keypoint_name", "x", "y", "conf"])

for i, frame_result in enumerate(pose_results):
        if frame_result.boxes is not None and frame_result.keypoints is not None:
            if frame_result.boxes.id is not None:
                track_ids = frame_result.boxes.id.cpu().numpy().astype(int)
                pts = frame_result.keypoints.xy.cpu().numpy()
                confs = frame_result.keypoints.conf.cpu().numpy()

                for pid_idx, track_id in enumerate(track_ids):
                    p_xy = pts[pid_idx]
                    p_conf = confs[pid_idx]
                    for kid, ((x, y), c) in enumerate(zip(p_xy, p_conf)):
                        name = keypoint_names[kid] if kid < len(keypoint_names) else f"kp_{kid}"
                        writer.writerow([i, track_id, kid, name, x, y, c])

print(f"Saved keypoint data to ({csv_path})")


# -----------------------------
# Step 6: Detect rackets
# -----------------------------
print("Running racket detection...")
racket_results = racket_model.track(
    source=str(video_path),
    save=False,
    verbose=True,
    conf=0.3
)

# -----------------------------
# Step 7: Save racket detections to CSV
# -----------------------------
racket_csv_path = output_dir / "racket_detections.csv"
print(f"Saving racket detections to: {racket_csv_path}")

with open(racket_csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "x1", "y1", "x2", "y2", "conf", "class_id"])

    for i, frame_result in enumerate(racket_results):
        if frame_result.boxes is not None:
            boxes = frame_result.boxes.xyxy.cpu().numpy()
            confs = frame_result.boxes.conf.cpu().numpy()
            class_ids = frame_result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = box
                writer.writerow([i, x1, y1, x2, y2, conf, class_id])

print(f"Saved racket detection data to ({racket_csv_path})")
