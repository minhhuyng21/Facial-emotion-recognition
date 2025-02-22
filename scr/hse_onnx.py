from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
import cv2
import numpy as np
from facenet_pytorch import MTCNN
import torch 
import time
def check_cuda():
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'
    return device 

def detect_face(frame, mtcnn):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bounding_boxes = []
    try:
        bounding_boxes, probs = mtcnn.detect(frame_rgb, landmarks=False)
        if probs is not None:
            bounding_boxes = bounding_boxes[probs > 0.9]  # Lọc khuôn mặt có độ tin cậy cao
    except Exception as e:
        print(f"Error detecting face: {e}")
    return (frame_rgb, bounding_boxes)

def display_prediction(frame_bgr, x1, y1, x2, y2, emotion):
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame_bgr, emotion, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    pass

def facial_emotion_recognition(bounding_boxes, frame_rgb, frame_bgr, emotion_responses, fer,time):
    if type(bounding_boxes) != type(None):
        for bbox in bounding_boxes:
            try:
                box = bbox.astype(int)
                x1, y1, x2, y2 = box[0:4]

                # Đảm bảo tọa độ không vượt quá kích thước ảnh
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)

                face_img = frame_rgb[y1:y2, x1:x2, :]
                # print(face_img.shape)
                emotion, scores = fer.predict_emotions(face_img)
                emotion_responses.append({
                        'time': time,
                        'emotion': emotion,
                        'score': scores 
                    })
                display_prediction(frame_bgr, x1, y1, x2, y2, emotion)
            except Exception as e:
                print(f"Error processing face: {e}")
    return emotion_responses

def display_fps(frame_bgr, prev_tick, fps):
    curr_tick = cv2.getTickCount()
    time_diff = (curr_tick - prev_tick) / cv2.getTickFrequency()
    fps = 1 / time_diff if time_diff > 0 else 0
    prev_tick = curr_tick
    cv2.putText(frame_bgr, f"FPS: {fps:.2f}", (5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
    return (prev_tick, fps)

def realtime_run(mtcnn, fer):
    emotion_responses = []    
    cap = cv2.VideoCapture(0)
    prev_tick = cv2.getTickCount()
    fps = 0

    start_time = time.time()
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb, bounding_boxes = detect_face(frame_bgr, mtcnn)
        elapsed_time = int(time.time() - start_time)
        emotion_responses = facial_emotion_recognition(bounding_boxes, frame_rgb, frame_bgr, emotion_responses, fer,elapsed_time) 
        prev_tick, fps = display_fps(frame_bgr, prev_tick, fps)
        cv2.imshow('Real-time Emotion Recognition', frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    return emotion_responses

def build(model_name):
    device = check_cuda()
    fer=HSEmotionRecognizer(model_name=model_name) #Facial emotion recognition 
    mtcnn = MTCNN(keep_all=False, post_process=False, min_face_size=40, device=device)  #Face detection
    return (fer, mtcnn)

if __name__ == "__main__":

    model_name='enet_b0_8_best_afew'
    # model_name = 'mobilevit_b0_va_mtl_7'
    # model_name='mobilenet_b0_7'

    fer, mtcnn = build(model_name=model_name)
    # face_img = cv2.imread('D:/hoasen/testing img/1.jpg')
    # emotion,scores=fer.predict_emotions(face_img=face_img,logits=False)

    emotion_responses = realtime_run(mtcnn, fer)
