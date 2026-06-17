import cv2
import numpy as np
import time
import csv
from datetime import datetime
from ultralytics import YOLO

model = YOLO("best.pt")
cap = cv2.VideoCapture("C://Users/user/Documents/videoplayback.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Employee Monitoring... Press 'q' to quit.")

log_file = "work_log.csv"
with open(log_file, 'w', newline='') as f:
    csv.writer(f).writerow(["Person ID", "Start Time", "End Time", "Total Work Time (s)"])

trackers, id_counter = {}, {}

def calculate_angle(a, b, c):
    """Calculate angle at point b given three points a, b, and c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def classify_posture(kps, standing_threshold=120, vertical_diff_threshold=0.3):
    try:
        # Keypoint indices for COCO format
        left_hip = kps[11][:2]
        left_knee = kps[13][:2]
        left_ankle = kps[15][:2]
        left_shoulder = kps[5][:2]

        # Calculate joint angles
        knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)

        # Normalize vertical difference between hip and knee (in image coordinates, y increases downwards)
        hip_y = left_hip[1]
        knee_y = left_knee[1]
        vertical_diff = (knee_y - hip_y) / (abs(hip_y) + 1e-6)

        # Heuristics for posture classification
        if knee_angle > standing_threshold and vertical_diff > vertical_diff_threshold:
            return "Standing"
        elif knee_angle < 100 and hip_angle < 120:
            return "Sitting"
        else:
            return "Unclear"
    except Exception as e:
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

def log_work(pid, start, end, total):
    with open(log_file, 'a', newline='') as f:
        csv.writer(f).writerow([pid,
            datetime.fromtimestamp(start).strftime("%Y-%m-%d %H:%M:%S"),
            datetime.fromtimestamp(end).strftime("%Y-%m-%d %H:%M:%S"),
            round(total, 2)
        ])

prev_kps = {}

while cap.isOpened():
    start_time = time.time()
    ret, frame = cap.read()
    if not ret: break

    now = time.time()
    results = model.predict(frame, imgsz=640, conf=0.5)[0]

    for idx, (kp_tensor, box) in enumerate(zip(results.keypoints.xy, results.boxes)):
        kps = kp_tensor.cpu().numpy()
        if len(kps) < 16: continue

        posture = classify_posture(kps)
        print (posture)
        pid = id_counter.setdefault(idx, len(id_counter))
        tracker = trackers.setdefault(pid, {"start": None, "work": 0.0, "last": now, "log": now})

        motion = detect_hand_motion(prev_kps.get(idx), kps)
        print(motion)
        if posture == "Standing":
            if motion:
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

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Time tracking
        if "Working" in label and posture == "Sitting":
            if tracker["start"] is None:
                tracker["start"] = now
                tracker["log"] = now
            elif now - tracker["log"] >= 60:
                tracker["work"] += now - tracker["start"]
                log_work(pid, now - tracker["work"], now, tracker["work"])
                tracker["start"], tracker["work"], tracker["log"] = now, 0.0, now
        elif tracker["start"] is not None:
            tracker["work"] += now - tracker["start"]
            log_work(pid, now - tracker["work"], now, tracker["work"])
            tracker["start"], tracker["work"], tracker["log"] = None, 0.0, now

        # Display ID and work time
        total_work = tracker["work"] + (now - tracker["start"] if tracker["start"] else 0)
        cv2.putText(frame, f"ID:{pid} Work:{int(total_work)}s", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        prev_kps[idx] = kps
        tracker["last"] = now

    # Remove stale trackers
    for pid in list(trackers):
        if now - trackers[pid]["last"] > 3:
            if trackers[pid]["start"]:
                trackers[pid]["work"] += now - trackers[pid]["start"]
                log_work(pid, now - trackers[pid]["work"], now, trackers[pid]["work"])
            del trackers[pid]

    # FPS and people count
    fps = 1.0 / (time.time() - start_time + 1e-6)
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Total People: {len(trackers)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Employee Work", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()