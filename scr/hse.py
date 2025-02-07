import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import torch
from hsemotion.facial_emotions import HSEmotionRecognizer
use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'

#Do not forget to run pip install facenet-pytorch
from facenet_pytorch import MTCNN
mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)

def detect_face(frame):
    bounding_boxes, probs = mtcnn.detect(frame, landmarks=False)
    bounding_boxes=bounding_boxes[probs>0.8]
    return bounding_boxes


model_name='enet_b0_8_best_afew'
# model_name='enet_b0_8_best_vgaf'
# model_name='enet_b0_8_va_mtl'
# model_name='enet_b2_8'

fer=HSEmotionRecognizer(model_name=model_name,device=device)
cap = cv2.VideoCapture(0)

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bounding_boxes = detect_face(frame_rgb)

    for bbox in bounding_boxes:
        box = bbox.astype(int)
        x1, y1, x2, y2 = box[0:4]

        # Cắt vùng khuôn mặt
        face_img = frame_rgb[y1:y2, x1:x2, :]

        # Dự đoán cảm xúc
        emotion, scores = fer.predict_emotions(face_img)

        # Vẽ bounding box và nhãn cảm xúc
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_bgr, emotion, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Hiển thị video
    cv2.imshow('Real-time Emotion Recognition', frame_bgr)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
