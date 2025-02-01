import requests
import os
from ultralytics import YOLO
import cv2
# Load a pretrained YOLO11n model
model = YOLO("assets/yolov11n-face.pt")
HUGGINGFACE = os.getenv("HUGGINGFACE")
API_URL = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"
headers = {"Authorization": f"Bearer {HUGGINGFACE}"}

def query_opencv(image):
    _, img_encoded = cv2.imencode(".jpg", image)  # Chuyển ma trận ảnh thành buffer
    response = requests.post(API_URL, headers=headers, data=img_encoded.tobytes())
    return response.json()

# output = query("DSC_6883.jpg")
# best_emotion = max(output, key=lambda x: x['score'])

frame = "statics/1000_F_506751155_fJ5Ko5T0wsTH7Q9VNwEgo6J81da8arlD.jpg"
frame = cv2.imread(frame)

results = model(frame)  # list of Results objects
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x, y, w, h = map(int, box.xyxy[0])
        face = frame[y:y + h, x:x + w]
        
        output = query_opencv(frame)

        best_emotion = max(output, key=lambda x: x['score'])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(frame, best_emotion['label'], (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        

cv2.imshow("Detected Faces", frame)  # Hiển thị ảnh với tiêu đề "Detected Faces"
cv2.waitKey(0)  # Chờ phím bất kỳ để đóng cửa sổ
cv2.destroyAllWindows()  # Đóng tất cả cửa sổ OpenCV
# print(best_emotion['label'])  # Kết quả: neutral
# print(output)