import streamlit as st
import asyncio
import threading
import multiprocessing
import cv2
import time
from scr.hse_onnx import build, detect_face, facial_emotion_recognition, display_fps    
from scr.data_draw import draw_diagram, save_diagram2pdf
# from scr.autogen_agent import expert_debate, data_analyze
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
    if 'visualize_button' not in st.session_state:
        st.session_state['visualize_button'] = {'button_state': False, 'text': ''}
    if 'chart' not in st.session_state:
        st.session_state['chart'] = {
                                    'line': None,
                                    'pie': None,
                                    'heat': None,
                                    'time': None,
                                        }

def stream_me():
    cap = cv2.VideoCapture(0)
    mtcnn = st.session_state['models']['mtcnn']
    fer = st.session_state['models']['fer']
    fps = 0
    prev_tick = cv2.getTickCount()
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb, bounding_boxes = detect_face(frame, mtcnn=mtcnn)
        elapsed_time = int(time.time() - start_time)
        st.session_state['emotion_responses'] = facial_emotion_recognition(bounding_boxes, frame_rgb, frame_bgr=frame, emotion_responses=st.session_state['emotion_responses'], fer=fer, time=elapsed_time)
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
        visualize_button_state = st.session_state['visualize_button']['button_state']
        if st.button('Visualize data'):
            if not visualize_button_state:
                st.session_state['visualize_button']['button_state'] = True
            else:
                st.session_state['visualize_button']['button_state'] = False
            line, pie, heat, time_chart = draw_diagram(emotion_responses)
            st.session_state['chart']['line'] = line
            st.session_state['chart']['pie'] = pie
            st.session_state['chart']['heat'] = heat
            st.session_state['chart']['time'] = time_chart    
        if st.session_state['visualize_button']['button_state']:           
            line = st.session_state['chart']['line']
            pie = st.session_state['chart']['pie']
            heat = st.session_state['chart']['heat']
            time_chart = st.session_state['chart']['time']
            st.pyplot(st.session_state['chart']['line'])
            st.pyplot(st.session_state['chart']['pie'])
            st.pyplot(st.session_state['chart']['heat'])
            st.pyplot(st.session_state['chart']['time'])
            # count, avg_scores = data_analyze(emotion_responses)
            # text  = expert_debate([count, avg_scores], "emotion.pdf")
            # st.write(text)
            if st.button('Export to PDF'):
                # save_diagram2pdf(line, pie, heat, time_chart, 'emotions.pdf')
                st.write('Save successfully')

    pass
if __name__ == '__main__':
    model_name = 'enet_b0_8_best_afew'
    setup_session_state(model_name=model_name)
    # print(st.session_state['models']['fer'])
    # print(st.session_state['models']['mtcnn'])
    main()
