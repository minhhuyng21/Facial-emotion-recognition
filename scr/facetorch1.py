from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
import torch
import time
import cv2
import json
path_img_input="D:\AI\Facial-emotion-recognition\config.yml"
path_img_output="test_output.jpg"
path_config="config.yml"


cfg = OmegaConf.load(path_config)
analyzer = FaceAnalyzer(cfg.analyzer)
emotion_responses = [] 
cap = cv2.VideoCapture(0)



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
# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()  # Capture frame-by-frame
    if not ret:
        print("Failed to grab frame.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Chuyển frame thành tensor kiểu uint8
    frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()  # [C, H, W]
    frame_tensor = frame_tensor.to(torch.uint8)  # Đảm bảo kiểu dữ liệu uint8


    response = analyzer.run(
        image_source=frame_tensor,
        batch_size=8,
        fix_img_size=False,
        return_img_data=True,
        include_tensors=False,
        path_output=None  # No output file needed
    )
    for face in response.faces:
        x1, y1, x2, y2 = face.loc.x1, face.loc.y1, face.loc.x2, face.loc.y2
        emotion = face.preds["fer"].label

        # Draw rectangle around the face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put emotion label above the rectangle
        cv2.putText(frame, emotion, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the real-time frame with detected emotions
    cv2.imshow("Real-Time Emotion Detection", frame)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()