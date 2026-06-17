def classify_posture(kps):
    hip = np.array(kps[11][:2])
    knee = np.array(kps[13][:2])
    ankle = np.array(kps[15][:2])
    shoulder = np.array(kps[5][:2])

    knee_angle = compute_knee_angle(hip, knee, ankle)

    hip_y = hip[1]
    knee_y = knee[1]
    ankle_y = ankle[1]
    shoulder_y = shoulder[1]

    torso_vec = hip - shoulder
    vertical = np.array([0, 1])
    torso_angle = np.degrees(np.arccos(
        np.clip(np.dot(torso_vec, vertical) / (np.linalg.norm(torso_vec) + 1e-6), -1.0, 1.0)
    ))

    hip_to_ankle = ankle_y - hip_y

    if knee_angle < 100 and hip_to_ankle > 50 and torso_angle < 35:
        return "Sitting"
    elif knee_angle >= 160 and hip_y < shoulder_y:
        return "Standing"
    elif hip_y >= shoulder_y and knee_angle >= 100:
        return "Floor sitting / legs extended"
    elif knee_angle < 70 and torso_angle > 40:
        return "Crouching"
    else:
        return "Uncertain"
