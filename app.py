import cv2
import asyncio
import aiohttp
import numpy as np
import os
from ultralytics import YOLO

# Load model YOLO (chạy trên GPU nếu có)
model = YOLO("assets/yolov11n-face.pt")
HUGGINGFACE = os.getenv("HUGGINGFACE")
API_URL = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE}"}

async def query_async(session, face_img):
    """Gửi ảnh khuôn mặt lên Hugging Face API bằng asyncio + aiohttp."""
    _, img_encoded = cv2.imencode(".jpg", face_img)
    async with session.post(API_URL, headers=HEADERS, data=img_encoded.tobytes()) as response:
        return await response.json()

async def process_faces(frame, session):
    """Phát hiện khuôn mặt và gửi API song song."""
    results = model(frame, verbose=False)  # Tắt log của YOLO để tăng tốc
    tasks = []
    faces = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Giảm kích thước khuôn mặt trước khi gửi API (tăng tốc)
            face_resized = cv2.resize(face, (64, 64))  

            faces.append((x1, y1, x2, y2))  # Lưu bounding box
            tasks.append(query_async(session, face_resized))  # Gửi API

    responses = await asyncio.gather(*tasks) if tasks else []
    return faces, responses

async def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    frame_count = 0  # Đếm số frame
    async with aiohttp.ClientSession() as session:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            if frame_count % 5 == 0:  # Chỉ gửi API mỗi 5 frame để tránh lag
                faces, responses = await process_faces(frame, session)
            else:
                faces, responses = [], []  # Không gửi API ở frame này

            # Vẽ kết quả lên ảnh
            for (x1, y1, x2, y2), output in zip(faces, responses):
                if isinstance(output, list) and len(output) > 0:
                    best_emotion = max(output, key=lambda x: x['score'])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, best_emotion['label'], (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

asyncio.run(main())  # Chạy chương trình async
