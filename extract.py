import cv2
import mediapipe as mp
import os
import csv
import glob

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)

DATA_PATH = "data/correct"
OUTPUT_PATH = "data/extracted"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Scan recursively for all .mp4 files
video_files = glob.glob(os.path.join(DATA_PATH, "**/*.mp4"), recursive=True)

for video_path in video_files:
    filename = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    csv_filename = os.path.splitext(filename)[0] + ".csv"
    csv_path = os.path.join(OUTPUT_PATH, csv_filename)

    print(f"xử lý {filename} ...")

    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        header = ["frame"]
        for i in range(33):
            header += [f"x_{i}", f"y_{i}"]
        writer.writerow(header)

        frame_idx = 0
        while True:
            success, frame = cap.read()
            if not success:
                break

            frame_idx += 1
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                row = [frame_idx]
                for lm in landmarks:
                    row.extend([lm.x, lm.y])
                writer.writerow(row)

    cap.release()
    print(f"lưu vào: {csv_path}")

pose.close()
print("\nDone extracting all videos!")
