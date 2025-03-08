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

# Khởi tạo cửa sổ OpenCV với kích thước Full HD
cv2.namedWindow("Squat Detection", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Squat Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cap = cv2.VideoCapture(0)

# Đặt độ phân giải của camera thành 1920x1080 (nếu camera hỗ trợ)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

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
            features.extend([landmarks[landmark].x, landmarks[landmark].y, landmarks[landmark].z, landmarks[landmark].visibility])
        
        # Chuyển đổi thành numpy array và chuẩn hóa
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        
        # Dự đoán bằng model
        prediction_probs = model.predict_proba(features)  # Trả về mảng xác suất
        label = np.argmax(prediction_probs)  # Lấy nhãn có xác suất cao nhất
        
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
        
        cv2.putText(frame, f"Prediction: {label_text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
    cv2.imshow("Squat Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

