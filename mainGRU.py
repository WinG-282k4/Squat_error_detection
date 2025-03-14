import cv2
import mediapipe as mp
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# Định nghĩa và đăng ký lớp Attention
@tf.keras.utils.register_keras_serializable(package='Custom', name='Attention')
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x * a
        return K.sum(output, axis=1)

# Load model Keras và scaler
scaler_path = "Model/scaler_GRU.pkl"
model_path = "Model/Squat_detection_GRU.keras"

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

model = load_model(model_path)  # Load model Keras

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
        
        # Lấy tọa độ x, y, z và độ tin cậy của keypoints quan trọng
        features = []
        for kp in IMPORTANT_KP:
            landmark = getattr(mp_pose.PoseLandmark, kp)
            features.extend([landmarks[landmark].x, landmarks[landmark].y, landmarks[landmark].z, landmarks[landmark].visibility])
        
        # Chuyển đổi thành numpy array và chuẩn hóa
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        
        # Thêm một chiều để phù hợp với định dạng đầu vào của mô hình
        features = np.expand_dims(features, axis=1)
        
        # Dự đoán bằng model Keras
        probabilities = model.predict(features)  # Lấy xác suất của từng lớp
        label = np.argmax(probabilities)  # Lấy nhãn có xác suất cao nhất
        
        # Nhãn lỗi Squat
        labels_dict = {
            0: "Correct",
            1: "Chan qua hep",
            2: "Chan qua rong",
            3: "Goi qua hep",
            4: "Xuong qua sau",
            5: "Lung gap"
        }
        label_text = labels_dict.get(label, "Unknown")
        
        # Hiển thị nhãn lên video
        cv2.putText(frame, f"Prediction: {label_text}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        
        # In xác suất của từng lớp
        print(f"Frame: {label_text}, Xác suất: {probabilities}")
    
    cv2.imshow("Squat Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()