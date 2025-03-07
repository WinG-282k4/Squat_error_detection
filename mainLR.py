import cv2
import mediapipe as mp
import numpy as np
import pickle

# Danh sách keypoints quan trọng
IMPORTANT_KP = [
    "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP", 
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
]

# Load mô hình và bộ chuẩn hóa
with open("Model/Squat_detection_LR.pkl", "rb") as f:
    model = pickle.load(f)
with open("Model/scaler_LR.pkl", "rb") as f:
    scaler = pickle.load(f)

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Khởi tạo OpenCV
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Chuyển đổi sang RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Lấy tọa độ x, y và độ tin cậy của keypoints quan trọng
        features = []
        for kp in IMPORTANT_KP:
            landmark = getattr(mp_pose.PoseLandmark, kp)
            features.extend([landmarks[landmark].x, landmarks[landmark].y, landmarks[landmark].visibility])
        
        # Chuyển đổi thành numpy array và chuẩn hóa
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        
        # Dự đoán bằng model
        prediction = model.predict(features)
        label = np.argmax(prediction)  # Lấy nhãn có xác suất cao nhất
        
        # Hiển thị kết quả lên màn hình
        labels_dict = {
            0: "Correct",
            1: "Chân quá hẹp",
            2: "Chân quá rộng",
            3: "Khoảng cách giữa 2 đầu gối quá nhỏ",
            4: "Xuống quá sâu",
            5: "Lỗi gập gập lưng"
        }
        label_text = labels_dict.get(label, "Unknown")
        
        cv2.putText(frame, f"Prediction: {label_text}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Squat Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
