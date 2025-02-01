import cv2
from deepface import DeepFace
import requests
import os
import numpy as np
from ultralytics import YOLO
# Load a pretrained YOLO11n model
model = YOLO("assets/yolov11n-face.pt")
HUGGINGFACE = os.getenv("HUGGINGFACE")
API_URL = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"
headers = {"Authorization": f"Bearer {HUGGINGFACE}"}

def query(image):
    _, img_encoded = cv2.imencode(".jpg", image)  # Chuyển ảnh thành buffer
    response = requests.post(API_URL, headers=headers, data=img_encoded.tobytes())
    return response.json()

cap = cv2.VideoCapture(0)
face_model = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    

    results = model(frame)  # list of Results objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, "mat lon", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị frame
    cv2.imshow("YOLOv8 Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()