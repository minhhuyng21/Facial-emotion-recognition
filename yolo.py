import requests
import os


HUGGINGFACE = os.getenv("HUGGINGFACE")
API_URL = "https://api-inference.huggingface.co/models/dima806/facial_emotions_image_detection"
headers = {"Authorization": f"Bearer {HUGGINGFACE}"}

def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

output = query("DSC_6883.jpg")
best_emotion = max(output, key=lambda x: x['score'])

print(best_emotion['label'])  # Kết quả: neutral
print(output)