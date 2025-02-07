from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
import torch
import time
import cv2

path_img_input="statics\\1000_F_506751155_fJ5Ko5T0wsTH7Q9VNwEgo6J81da8arlD.jpg"
path_img_output="test_output.jpg"
path_config="config.yml"


cfg = OmegaConf.load(path_config)
analyzer = FaceAnalyzer(cfg.analyzer)

cap = cv2.VideoCapture(0)

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


    start_time = time.time()
    analyzer
    response = analyzer.run(
        image_source=frame_tensor,
        batch_size=8,
        fix_img_size=False,
        return_img_data=True,
        include_tensors=False,
        path_output=None  # No output file needed
    )
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Thời gian xử lý: {processing_time:.4f} giây")
    # Draw bounding boxes and emotion labels on the frame
    start_time = time.time()
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
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Thời gian xử lý: {processing_time:.4f} giây")
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close OpenCV window
cap.release()
cv2.destroyAllWindows()