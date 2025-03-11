import cv2
import mediapipe as mp
import pandas as pd

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Danh sách các keypoint quan trọng
Important_kp = [
    mp_pose.PoseLandmark.NOSE,
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

# DataFrame để lưu dữ liệu
columns = ['label'] + [f"{kp.name}_{coord}" for kp in Important_kp for coord in ['x', 'y', 'z', 'visibility']] + ['video_name']
df = pd.DataFrame(columns=columns)

# Hàm xử lý video và gán nhãn
def label_video(video_path):
    cap = cv2.VideoCapture(video_path)
    label = None

    def set_label(event, x, y, flags, param):
        nonlocal label
        if event == cv2.EVENT_LBUTTONDOWN:
            if 49 <= x <= 53:  # ASCII code for keys 1 to 5
                label = x - 48

    cv2.namedWindow('Video')
    cv2.setMouseCallback('Video', set_label)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            row = []
            for kp in Important_kp:
                keypoint = landmarks[kp.value]
                row.extend([keypoint.x, keypoint.y, keypoint.z, keypoint.visibility])

            if label is not None:
                row = [label] + row + [video_path.split('/')[-1]]
                df.loc[len(df)] = row
                label = None

            # Vẽ keypoints lên frame
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Video', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Gọi hàm để xử lý video
label_video('path_to_your_video.mp4')

# Lưu DataFrame vào file CSV
df.to_csv('labeled_data.csv', index=False)