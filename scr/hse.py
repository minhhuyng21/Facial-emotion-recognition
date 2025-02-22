import os
import numpy as np
import cv2
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer
from facenet_pytorch import MTCNN
import json
from data_draw import draw_diagram
from autogen_agent import expert_debate
import pandas as pd
import time
# Kiểm tra GPU
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
emotion_responses = []
# Khởi tạo mô hình phát hiện khuôn mặt
mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

def data_analyze(emotion_respose):
    df = pd.DataFrame([{'emotion': item['emotion']} for item in emotion_responses])

    # Đếm số lượng của mỗi emotion
    emotion_counts = df['emotion'].value_counts()

    scores_df = pd.DataFrame([item['score'] for item in emotion_responses])
    scores_df.columns = [f'Score_{i+1}' for i in range(8)]
    scores_df['Emotion'] = [item['emotion'] for item in emotion_responses]

    # Tính trung bình score cho mỗi emotion
    avg_scores = scores_df.groupby('Emotion')[scores_df.columns[:-1]].mean()

    return emotion_counts, avg_scores

 
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Tránh tràn số
    return exp_x / np.sum(exp_x)

def save_emotions():
    with open('emotions.json', 'w') as json_file:
        json.dump(emotion_responses, json_file, indent=4)
# Check if the webcam is opened correctly
# Hàm phát hiện khuôn mặt với xử lý lỗi
def detect_face(frame):
    try:
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs is not None:
            bounding_boxes = bounding_boxes[probs > 0.9]  # Lọc khuôn mặt có độ tin cậy cao
        else:
            bounding_boxes = []
    except Exception as e:
        print(f"Error detecting face: {e}")
        bounding_boxes = []
    return bounding_boxes

# Khởi tạo mô hình nhận diện cảm xúc
model_name = 'enet_b0_8_best_afew'
fer = HSEmotionRecognizer(model_name=model_name, device=device)

cap = cv2.VideoCapture(0)
prev_tick = cv2.getTickCount()
fps = 0

start_time = time.time()
while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bounding_boxes = detect_face(frame_rgb)

    if len(bounding_boxes) > 0:
        for bbox in bounding_boxes:
            try:
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]

                # Đảm bảo tọa độ không vượt quá kích thước ảnh
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)

                # Cắt vùng khuôn mặt
                face_img = frame_rgb[y1:y2, x1:x2, :]

                # Dự đoán cảm xúc
                emotion, scores = fer.predict_emotions(face_img)
                elapsed_time = int(time.time() - start_time)  # Tính số giây từ đầu buổi học
                emotion_responses.append({
                    'time_in_class': elapsed_time,  # Lưu giây thứ mấy của buổi học
                    'emotion': emotion,
                    'score': softmax(scores).tolist()
                })

                # Vẽ bounding box và nhãn cảm xúc
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_bgr, emotion, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            except Exception as e:
                print(f"Error processing face: {e}")

    # Hiển thị video
    curr_tick = cv2.getTickCount()
    time_diff = (curr_tick - prev_tick) / cv2.getTickFrequency()
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_tick = curr_tick
    cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('Real-time Emotion Recognition', frame_bgr)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

draw_diagram(emotion_responses,"emotion.pdf")
# count, score = data_analyze(emotion_responses)
# expert_debate([count,score],"emotion.pdf")

