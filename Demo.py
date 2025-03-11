import cv2
import mediapipe as mp
import numpy as np
import pickle

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

# Load mô hình và bộ chuẩn hóa
with open("Model/Squat_detection_LR.pkl", "rb") as f:
    model = pickle.load(f)
with open("Model/scaler_LR.pkl", "rb") as f:
    scaler = pickle.load(f)

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Đọc video đầu vào
# video_path = "Data/Test/Chan_qua_hep/20250228_084059000_iOS.mp4"  
# video_path = "Loi_gap_lung.mp4"
video_path = "Demo/Video_demo.mp4"
cap = cv2.VideoCapture(video_path)

# Lấy thông tin video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Tạo VideoWriter để lưu video đầu ra
output_path = "Demo/output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Định dạng MP4
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
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
        
        # Chuyển đổi thành numpy array và chuẩn hóa
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        
        # Dự đoán bằng model
        # prediction = model.predict(features)[0]

        # label = int(prediction)  # ✅ Đảm bảo label là số nguyên

        prediction_probs = model.predict_proba(features)  # Trả về mảng xác suất
        label = np.argmax(prediction_probs)  # Lấy nhãn có xác suất cao nhất
        
        # Nhãn lỗi Squat
        labels_dict = {
            0: "Correct",
            1: "Chan qua hep",
            2: "Chan qua rong",
            3: "Goi qua hep ",
            4: "Xuong qua sau",
            5: "Lung gap"
        }
        label_text = labels_dict.get(label, "Unknown")
        
        # Hiển thị nhãn lên video
        cv2.putText(frame, f"Prediction: {label_text}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    
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
