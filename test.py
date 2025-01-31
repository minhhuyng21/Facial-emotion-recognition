from ultralytics import YOLO
# Load a pretrained YOLO11n model
model = YOLO("assets/yolov11n-face.pt")

# Define path to the image file
source = "statics/1000_F_506751155_fJ5Ko5T0wsTH7Q9VNwEgo6J81da8arlD.jpg"

# Run inference on the source
results = model(source)  # list of Results objects

for result in results:
    boxes = result.boxes.xywh# Boxes object for bounding box outputs
boxes = boxes.tolist()
print(boxes)