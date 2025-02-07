from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
from torch.nn.functional import cosine_similarity
from typing import Dict
import cv2

path_img_input="statics\\1000_F_506751155_fJ5Ko5T0wsTH7Q9VNwEgo6J81da8arlD.jpg"
path_img_output="test_output.jpg"
path_config="config.yml"


cfg = OmegaConf.load(path_config)
analyzer = FaceAnalyzer(cfg.analyzer)

image_path = "statics\\1000_F_506751155_fJ5Ko5T0wsTH7Q9VNwEgo6J81da8arlD.jpg"  # Replace with your image path
image = cv2.imread(image_path)

# Run the FaceAnalyzer without saving the output
response = analyzer.run(
    path_image=image_path,
    batch_size=8,
    fix_img_size=False,
    return_img_data=True,  # Return image data to work with OpenCV
    include_tensors=False,
    path_output=None  # No output file
)

# Draw bounding boxes and emotion labels on the image
for face in response.faces:
    # Get face coordinates
    x1, y1, x2, y2 = face.loc.x1, face.loc.y1, face.loc.x2, face.loc.y2

    # Get emotion label
    emotion = face.preds["fer"].label

    # Draw rectangle around the face
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Put the emotion label above the rectangle
    cv2.putText(image, emotion, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with OpenCV
cv2.imshow("Emotion Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()