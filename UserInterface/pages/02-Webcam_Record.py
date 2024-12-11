import cv2
import os
import time
import json
import streamlit as st
import numpy as np
import sys
sys.path.append(os.getcwd())
from MediaLoader.utils import encode_video_H264, get_video_info_dict, cut_video
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown, init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.registry import VISUALIZERS

# The output file path
output_file = os.path.join('UserInterface', 'outputs', 'visualizations', 'webcam.mp4')

# Parameters of the video writer
camera = cv2.VideoCapture(0)
frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'avc1')

################## Functions ##################

def record_video():
    '''Start recording the webcam feed and save it to a file'''
    st.session_state.recording = True
    fps = 5
    st.session_state.video_writter = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

def stop_video():
    '''Stop recording the webcam feed and save it to a file'''

    if st.session_state.recording:
        st.session_state.recording = False
        st.session_state.video_writter.release()
        camera.release()
        cv2.destroyAllWindows()

        if pose_sequence:
            st.session_state.video_path = output_file
        with open(os.path.join('UserInterface', 'outputs', 'predictions', 'webcam') + '.json', 'w') as f:
                json.dump(pose_sequence, f)
    
    else:
        print("You are not recording")

def remove_first_lines_from_json_file(json_file_path, num_lines_to_remove):
    '''Remove the first lines of a json file'''
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    data = data[num_lines_to_remove-1:]
    with open(json_file_path, 'w') as f:
        json.dump(data, f)

def remove_last_lines_from_json_file(json_file_path, num_lines_to_remove):
    '''Remove the last lines of a json file'''
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    data = data[:-num_lines_to_remove]
    with open(json_file_path, 'w') as f:
        json.dump(data, f)

def remove_frames(start_frame, end_frame):
    json_file_path = st.session_state.video_path.replace('.mp4', '.json').replace('visualizations', 'predictions')
    if start_frame != 0:
        remove_first_lines_from_json_file(json_file_path, start_frame)
    if end_frame != video_dic['num_frames']:
        remove_last_lines_from_json_file(json_file_path, video_dic['num_frames'] - end_frame)
    output_path = st.session_state.video_path.replace('.mp4', '_cut.mp4')
    os.rename(json_file_path, json_file_path.replace('.json', '_cut.json'))
    cut_video(st.session_state.video_path, start_frame-1, end_frame, output_path)
    os.remove(st.session_state.video_path)
    st.session_state.video_path = output_path

def remove_video():
    '''Remove the video file and the predictions file'''
    os.remove(st.session_state.video_path)
    os.remove(st.session_state.video_path.replace('.mp4', '.json').replace('visualizations', 'predictions'))
    st.session_state.video_path = None


################## Session state ##################

if 'video_writter' not in st.session_state:
    st.session_state.video_writter = None

if 'recording' not in st.session_state:
    st.session_state.recording = False

# Store the inferencer in the session state
if 'inferencer' not in st.session_state:
    st.session_state['inferencer'] = None

# Store the video path in the session state, this path is used with the 'Webcam' page only
if 'video_path' not in st.session_state:
    st.session_state.video_path = None


################## Main ##################

st.title(":camera: Webcam Live Feed")

col1, col2, col3 = st.columns(3)

with col1:
    start = st.button(':black_circle_for_record: Start Recording', on_click=record_video)
with col2:
    stop = st.button(':black_square_for_stop: Stop Recording', on_click=stop_video)


# Display the webcam feed
FRAME_WINDOW = st.image([])
num_frames = 0
start_time = time.time()

# to save the prediction
pose_sequence = []

if st.session_state['inferencer'] is not None:
    while st.session_state.recording:
        _, frame = camera.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_dict = {}
        frame_dict['frame_id'] = num_frames

        # topdown pose estimation
        preds = st.session_state['inferencer'].process_one_frame(frame, output_directory='UserInterface/outputs/')
        frame_dict['instances'] = preds
        
        frame = st.session_state['inferencer'].visualizer.get_image()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        num_frames += 1
        fps = num_frames / (time.time() - start_time)

        # Add text
        size_text = 0.5
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, size_text, (255, 0, 0))
        cv2.putText(frame, f"Resolution: {frame_height}x{frame_width}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, size_text, (255, 0, 0))
        cv2.putText(frame, f"Frame: {num_frames}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, size_text, (255, 0, 0))

        FRAME_WINDOW.image(frame)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        st.session_state.video_writter.write(frame_bgr)
        pose_sequence.append(frame_dict)
else:
    st.write(':heavy_exclamation_mark: **WARNING** : You need to load a *Pose Estimation model* first :heavy_exclamation_mark:')

# Display the video saved
if st.session_state.video_path is not None:
    st.write('## Video saved :')
    col1, col2, col3 = st.columns(3)
    video_dic = get_video_info_dict(st.session_state.video_path)
    with col1:
        start_frame = st.slider("Start Frame", 1, video_dic['num_frames'], 1)
    with col2:
        end_frame = st.slider("End Frame", start_frame, video_dic['num_frames'], video_dic['num_frames'])
    with col3:
        st.button('Apply changes', on_click=remove_frames, args=(start_frame, end_frame))

    st.video(st.session_state.video_path)
    print(get_video_info_dict(st.session_state.video_path))

    col1, col2, col3 = st.columns(3)
    with col3:
        delete_button = st.button('Delete video', on_click=remove_video, )

print(st.session_state)
