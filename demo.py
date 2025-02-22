import streamlit as st
import asyncio
import threading
import multiprocessing
import cv2

from scr.hse_onnx import build, detect_face, facial_emotion_recognition, display_fps    
from scr.data_draw import draw_diagram, save_diagram2pdf

@st.cache_resource
def setup(model_name):
    fer, mtcnn = build(model_name)
    return (fer, mtcnn)

def setup_session_state(model_name):
    if 'emotion_responses' not in st.session_state:
        st.session_state['emotion_responses'] = []    
    if 'realtime_button' not in st.session_state:
        st.session_state['realtime_button'] = {'is_realtime': False, 'text': ''}
    if 'models' not in st.session_state:
        fer, mtcnn = setup(model_name=model_name)
        st.session_state['models'] = {'fer': fer, 'mtcnn': mtcnn}

def stream_me():
    cap = cv2.VideoCapture(0)
    mtcnn = st.session_state['models']['mtcnn']
    fer = st.session_state['models']['fer']
    fps = 0
    prev_tick = cv2.getTickCount()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb, bounding_boxes = detect_face(frame, mtcnn=mtcnn)
        st.session_state['emotion_responses'] = facial_emotion_recognition(bounding_boxes, frame_rgb, frame_bgr=frame, emotion_responses=st.session_state['emotion_responses'], fer=fer)
        prev_tick, fps = display_fps(frame, prev_tick, fps)
        st.image(frame, channels='BGR')
        if st.session_state['realtime_button']['is_realtime'] == False:
            break
    pass

def main():
    realtime_button_state = st.session_state['realtime_button']['is_realtime']
    if not realtime_button_state:
        if st.button('Run realtime', type='primary'):
            st.session_state['realtime_button']['is_realtime'] = True
            st.session_state['realtime_button']['text'] = ''
            st.rerun()
    else:
        if st.button('Exit realtime', type='secondary'):
            st.session_state['realtime_button']['is_realtime'] = False
            st.session_state['realtime_button']['text'] = ''
            st.rerun()

    if realtime_button_state:
        with st.empty():
            stream_me()

    emotion_responses = st.session_state['emotion_responses']
    if  emotion_responses:
        if st.button('Visualize data'):
            line, pie, heat = draw_diagram(emotion_responses)
            st.pyplot(line)
            st.pyplot(pie)
            st.pyplot(heat)
            if st.button('Export to PDF'):
                save_diagram2pdf(line, pie, heat, 'emotions.pdf')

    pass
if __name__ == '__main__':
    model_name = 'enet_b0_8_best_afew'
    setup_session_state(model_name=model_name)
    # print(st.session_state['models']['fer'])
    # print(st.session_state['models']['mtcnn'])
    main()
