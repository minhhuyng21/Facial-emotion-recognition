import cv2
from deepface import DeepFace
import requests
import os
import numpy as np
from ultralytics import YOLO
import asyncio
import aiohttp
import json


# Load a pretrained YOLO11n model
model = YOLO("assets/yolov11n-face.pt")
HUGGINGFACE = os.getenv("HUGGINGFACE")
API_URL = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE}"}
emotion_responses = [] 

async def query_async(session, face_img):
    """Gửi ảnh khuôn mặt lên Hugging Face API bằng asyncio + aiohttp."""
    _, img_encoded = cv2.imencode(".jpg", face_img)
    async with session.post(API_URL, headers=HEADERS, data=img_encoded.tobytes()) as response:
        return await response.json()

async def process_frame(frame):
    """Phát hiện khuôn mặt và gửi API song song."""
    results = model(frame)
    tasks = []
    faces = []

    async with aiohttp.ClientSession() as session:
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                face = frame[y1:y2, x1:x2]  # Cắt ảnh khuôn mặt

                if face.size == 0:  # Kiểm tra nếu không có khuôn mặt hợp lệ
                    continue

                faces.append((x1, y1, x2, y2))  # Lưu bounding box
                tasks.append(query_async(session, face))  # Tạo task gửi API

        responses = await asyncio.gather(*tasks)  # Chạy các request API song song
    
    print("API Response:", responses)  # In ra dữ liệu trả về để kiểm tra


    for response in responses:
        emotion_responses.append(response)
    # save_emotions(responses)
    for (x1, y1, x2, y2), output in zip(faces, responses):
        if isinstance(output, list) and len(output) > 0:  # Đảm bảo API trả về kết quả hợp lệ
            best_emotion = max(output, key=lambda x: x['score'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, best_emotion['label'], (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame


def save_emotions():
    """Lưu các cảm xúc có score > 0.7 vào list"""

    print("****************************************")
    filtered_emotions = []
    
    for response in emotion_responses:
        for emotion in response:
            if emotion['score'] > 0.7:  # Chỉ chọn những cảm xúc có score > 0.7
                filtered_emotions.append(emotion)
    
    # Lưu vào tệp JSON sau khi hoàn tất (nếu cần)
    with open('emotions.json', 'w') as json_file:
        json.dump(filtered_emotions, json_file, indent=4)

def test():
    frame = cv2.imread("statics/DSC_6883.jpg")
    processed_frame = asyncio.run(process_frame(frame))  # Gọi async function
    cv2.imshow("Detected Faces", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        processed_frame = asyncio.run(process_frame(frame))  # Gọi async function
        cv2.imshow("YOLOv8 Face Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # print(emotion_responses)
    # save_emotions()


if __name__ == "__main__":
    main()