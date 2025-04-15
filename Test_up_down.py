import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# Tải cả mô hình và scaler
with open("model_UD/LR_Up_Down_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_UD/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Danh sách keypoints quan trọng
IMPORTANT_KP = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE"
]

# Create column names that match those used during training
headers = []
for kp in IMPORTANT_KP:
    headers.extend([f"{kp.lower()}_x", f"{kp.lower()}_y", f"{kp.lower()}_z", f"{kp.lower()}_v"])

#State
UP = 1
DOWN = 0

#Khởi tạo biến
count = 0
pre_state = -1

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Đọc video đầu vào
video_path = "Demo/Video_demo1.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy thông tin video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo VideoWriter để lưu video đầu ra
output_path = "Demo/Dem_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

#Lọc nhiễu
pre_time = datetime(1970, 1, 1, 0, 0, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print(f"Total count: {count}")
        count = 0
        break  # Thoát nếu hết video
    
    # Chuyển đổi sang RGB để xử lý với MediaPipe
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Lấy tọa độ x, y, z và độ tin cậy của keypoints quan trọng
        features = []
        for kp in IMPORTANT_KP:
            landmark = getattr(mp_pose.PoseLandmark, kp)
            features.extend([landmarks[landmark].x, landmarks[landmark].y, landmarks[landmark].z, landmarks[landmark].visibility])
        
        # Chuyển đổi thành DataFrame với tên cột tương ứng và chuẩn hóa
        features_array = np.array(features).reshape(1, -1)
        features_df = pd.DataFrame(features_array, columns=headers)
        features = scaler.transform(features_df)
        
        # Dự đoán bằng model Keras
        label_array = model.predict(features)  # Returns a NumPy array
        label = label_array[0]
        
        # Nhãn lỗi Squat
        labels_dict = {
            0: "Down",
            1: "Up",
        }
        label_text = labels_dict.get(label, "Unknown")

        # Lấy thời gian hiện tại
        now_time = datetime.now()
        if (now_time - pre_time).total_seconds() > 0.1:
            if label_text == "Down":
                if pre_state == UP and pre_state != -1:
                    count += 1
        
            pre_state = DOWN if label_text == "Down" else UP
            pre_time = now_time
        # Hiển thị nhãn lên video
        cv2.putText(frame, f"Prediction: {label_text}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(frame, f"count: {count}", (50, 150), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # In xác suất của từng lớp
        print(f"Frame: {label_text}")
    
    # Ghi frame có nhãn vào video output
    out.write(frame)
    
    # Hiển thị video trong quá trình xử lý
    cv2.imshow("Squat Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
out.release()
cv2.destroyAllWindows()