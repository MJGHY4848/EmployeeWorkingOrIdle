import cv2
import numpy as np
import time
import csv
from datetime import datetime
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO("best.pt")  # Replace with your model path

# Start webcam
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("C://Users/user/Documents/videoplayback.mp4")  #"rtsp://admin:uct_1122@192.168.0.2:554/Streaming/channels/101"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Employee Monitoring... Press 'q' to quit.")

previous_keypoints = {}

# Tracking
person_tracker = {}  # {id: {start_time, work_time, last_seen, last_logged_time}}
id_counter = 0
id_map = {}

# CSV setup
log_file = "work_log.csv"
with open(log_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Person ID", "Start Time", "End Time", "Total Work Time (s)"])

# Posture detection
def classify_posture(kps):
    try:
        hip = np.array(kps[11][:2])
        knee = np.array(kps[13][:2])
        ankle = np.array(kps[15][:2])
        vec1 = hip - knee
        vec2 = ankle - knee
        angle = np.degrees(np.arccos(
            np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-6), -1.0, 1.0)
        ))
        # Assumes all keypoints are available
        hip_y = kps[11][1]
        knee_y = kps[13][1]
        ankle_y = kps[15][1]
        shoulder_y = kps[5][1]

        # Knee angle as before
        knee_angle = angle

        # Vertical distance
        hip_to_ankle = ankle_y - hip_y
        hip_to_shoulder = shoulder_y - hip_y

        if knee_angle < 100:
            return "Sitting"
        elif knee_angle > 120:
            if hip_y > shoulder_y + 40:  # Adjust threshold for scale
                return "Sitting"
            elif hip_to_ankle < 50:  # hip close to ankle → standing tall
                return "Standing"
            else:
                return "Unknown"
        else:
            return "Unknown"

    except:
        pass
    return "Unknown"

# Hand movement
def detect_hand_motion(prev_kps, curr_kps, threshold=15):
    try:
        for idx in [9, 10]:
            prev = np.array(prev_kps[idx][:2])
            curr = np.array(curr_kps[idx][:2])
            if np.linalg.norm(curr - prev) > threshold:
                return True
    except:
        pass
    return False

# Main loop
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    results = model.predict(frame, imgsz=640, conf=0.5)
    keypoints_list = results[0].keypoints
    boxes = results[0].boxes
    now = time.time()

    if keypoints_list and boxes:
        for i, kp_tensor in enumerate(keypoints_list.xy):
            kp_np = kp_tensor.cpu().numpy()
            if kp_np.shape[0] < 16:
                continue

            posture = classify_posture(kp_np)
            hand_moving = detect_hand_motion(previous_keypoints.get(i), kp_np)

            if posture == "Standing":
                if hand_moving:
                    label = "Standing - Working"
                    color = (0, 255, 0)
                else:
                    label = "Standing - Not Working"
                    color = (0, 0, 255)
            elif posture == "Sitting":
                label = "Sitting - Working"
                color = (0, 255, 0)
            else:
                label = "Moving - Not Working"
                color = (0, 0, 139)

            # Draw box + label
            box = boxes[i]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Assign ID
            if i not in id_map:
                id_map[i] = id_counter
                id_counter += 1
            person_id = id_map[i]

            if person_id not in person_tracker:
                person_tracker[person_id] = {
                    "start_time": None,
                    "work_time": 0.0,
                    "last_seen": now,
                    "last_logged_time": now
                }

            tracker = person_tracker[person_id]
            tracker["last_seen"] = now

            # Work session tracking
            if label == "Sitting - Working":
                if tracker["start_time"] is None:
                    tracker["start_time"] = now
                    tracker["last_logged_time"] = now
                else:
                    if now - tracker["last_logged_time"] >= 60:  # 60 seconds
                        tracker["work_time"] += now - tracker["start_time"]
                        total_work_time = round(tracker["work_time"], 2)
                        start_time = datetime.fromtimestamp(now - total_work_time).strftime("%Y-%m-%d %H:%M:%S")
                        end_time = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")
                        with open(log_file, mode='a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([person_id, start_time, end_time, total_work_time])
                        tracker["start_time"] = now
                        tracker["work_time"] = 0.0
                        tracker["last_logged_time"] = now

            else:
                if tracker["start_time"] is not None:
                    tracker["work_time"] += now - tracker["start_time"]
                    total_work_time = round(tracker["work_time"], 2)
                    start_time = datetime.fromtimestamp(now - total_work_time).strftime("%Y-%m-%d %H:%M:%S")
                    end_time = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

                    with open(log_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([person_id, start_time, end_time, total_work_time])

                    tracker["start_time"] = None
                    tracker["work_time"] = 0.0
                    tracker["last_logged_time"] = now

            # Show ID and work time
            display_time = tracker["work_time"]
            if tracker["start_time"]:
                display_time += now - tracker["start_time"]

            cv2.putText(frame, f"ID:{person_id} Work:{int(display_time)}s", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            previous_keypoints[i] = kp_np

    # Handle person disappearing from camera
    for pid, data in list(person_tracker.items()):
        if now - data["last_seen"] > 3:
            if data["start_time"]:
                data["work_time"] += now - data["start_time"]
                total_work_time = round(data["work_time"], 2)
                start_time = datetime.fromtimestamp(now - total_work_time).strftime("%Y-%m-%d %H:%M:%S")
                end_time = datetime.fromtimestamp(now).strftime("%Y-%m-%d %H:%M:%S")

                with open(log_file, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([pid, start_time, end_time, total_work_time])

            del person_tracker[pid]

    cv2.putText(frame, f"Total People: {len(person_tracker)}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Employee Work", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
