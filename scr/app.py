import streamlit as st
import cv2
import time
import numpy as np
import torch
import pandas as pd
import json
from facenet_pytorch import MTCNN
from hsemotion.facial_emotions import HSEmotionRecognizer
from scr.data_draw import draw_diagram
from autogen_agent import expert_debate
import time

# Khởi tạo trạng thái phiên (session state)
if 'running' not in st.session_state:
    st.session_state.running = False
if 'emotion_responses' not in st.session_state:
    st.session_state.emotion_responses = []

# Kiểm tra GPU và thiết lập device
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

# Khởi tạo mô hình phát hiện khuôn mặt và nhận diện cảm xúc
mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)
model_name = 'enet_b0_8_best_afew'
fer = HSEmotionRecognizer(model_name=model_name, device=device)

# Hàm softmax
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Tránh tràn số
    return exp_x / np.sum(exp_x)

# Hàm phát hiện khuôn mặt
def detect_face(frame):
    try:
        bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
        if probs is not None:
            bounding_boxes = bounding_boxes[probs > 0.9]  # Lọc khuôn mặt có độ tin cậy cao
        else:
            bounding_boxes = []
    except Exception as e:
        message = st.empty()
        message.write("Không phát hiện khuôn mặt")
        time.sleep(1)  # Hiển thị thông báo trong 2 giây
        message.empty()  # Xóa thông báo
        bounding_boxes = []
    return bounding_boxes

# Hàm phân tích dữ liệu
def data_analyze(emotion_responses):
    df = pd.DataFrame([{'emotion': item['emotion']} for item in emotion_responses])
    emotion_counts = df['emotion'].value_counts()

    scores_df = pd.DataFrame([item['score'] for item in emotion_responses])
    scores_df.columns = [f'Score_{i+1}' for i in range(8)]
    scores_df['Emotion'] = [item['emotion'] for item in emotion_responses]

    avg_scores = scores_df.groupby('Emotion')[scores_df.columns[:-1]].mean()
    return emotion_counts, avg_scores

st.title("Real-time Emotion Recognition")

# Hiển thị 2 nút: "Show Realtime" và "Stop"
col1, col2 = st.columns(2)
if col1.button("Show Realtime"):
    st.session_state.running = True
if col2.button("Stop"):
    st.session_state.running = False

# Nếu trạng thái đang chạy thì bắt đầu stream video từ webcam
if st.session_state.running:
    st.info("Đang chạy realtime. Nhấn nút Stop để dừng.")
    frame_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    prev_tick = cv2.getTickCount()

    start_time = time.time()
    while st.session_state.running:
        ret, frame_bgr = cap.read()
        if not ret:
            st.write("Không thể truy cập camera.")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        bounding_boxes = detect_face(frame_rgb)

        if len(bounding_boxes) > 0:
            for bbox in bounding_boxes:
                try:
                    box = bbox.astype(int)
                    x1, y1, x2, y2 = box[0:4]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)
                    face_img = frame_rgb[y1:y2, x1:x2, :]

                    # Dự đoán cảm xúc và lưu kết quả
                    emotion, scores = fer.predict_emotions(face_img)
                    elapsed_time = int(time.time() - start_time)
                    st.session_state.emotion_responses.append({
                        'time': elapsed_time,
                        'emotion': emotion,
                        'score': softmax(scores).tolist()
                    })

                    # Vẽ bounding box và nhãn cảm xúc lên frame
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame_bgr, emotion, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                except Exception as e:
                    message = st.empty()
                    message.write("Không phát hiện khuôn mặt")
                    time.sleep(1)  # Hiển thị thông báo trong 2 giây
                    message.empty()  # Xóa thông báo

        curr_tick = cv2.getTickCount()
        time_diff = (curr_tick - prev_tick) / cv2.getTickFrequency()
        fps = 1 / time_diff if time_diff > 0 else 0
        prev_tick = curr_tick
        cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (5, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        frame_placeholder.image(frame_bgr, channels="BGR")
        time.sleep(0.03)

    cap.release()
    st.success("Đã dừng realtime capture.")

# Sau khi dừng realtime, nếu có dữ liệu cảm xúc thu thập được thì hiển thị các nút phân tích
if not st.session_state.running and st.session_state.emotion_responses:
    st.subheader("Phân tích dữ liệu")
    
    if st.button("Xem ảnh phân tích"):
        # Vẽ biểu đồ và lưu vào file PDF (bạn có thể điều chỉnh hàm draw_diagram để hiển thị ảnh nếu cần)
        draw_diagram(st.session_state.emotion_responses, "emotion.pdf")
        st.success("Đã tạo file PDF chứa biểu đồ phân tích (emotion.pdf).")
    
    if st.button("Xem phân tích dạng text"):
        count, avg_scores = data_analyze(st.session_state.emotion_responses)
        st.write("**Số lượng cảm xúc:**")
        st.write(count)
        st.write("**Điểm trung bình cho mỗi cảm xúc:**")
        st.write(avg_scores)
        # Gọi thêm hàm expert_debate để thêm phần thảo luận vào PDF nếu cần
        expert_debate([count, avg_scores], "emotion.pdf")
        st.success("Đã cập nhật thông tin phân tích vào file PDF.")
    
    # Nút xuất file PDF
    try:
        with open("emotion.pdf", "rb") as pdf_file:
            st.download_button(label="Xuất file PDF", data=pdf_file, file_name="emotion.pdf", mime="application/pdf")
    except Exception as e:
        st.write("Chưa tạo được file PDF để xuất. Vui lòng kiểm tra lại các bước phân tích.")
