import streamlit as st
import os
import cv2
import json
import sys
sys.path.append(os.getcwd())
from UserInterface.utils import add_logo, draw_bounding_box_and_label, colors, color_names
from UserInterface.pages.Analyzer import squat_analyzer

add_logo("UserInterface/HEADMIND PARTNERS AI -  Bleu.png", height=300)

#################### Session state ####################

if 'movement' not in st.session_state:
            st.session_state['movement'] = None

if 'selected_json_file' not in st.session_state:
    st.session_state['selected_json_file'] = None

#################### Main ####################

st.title(':woman-cartwheeling: Movement Analyzer')

folder_path = 'UserInterface/outputs/visualizations'
video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
selected_video_file = st.sidebar.selectbox('Choose a video file', video_files)

if selected_video_file is not None:

    file_name = selected_video_file.split('.')[0]
    json_file_path = os.path.join('UserInterface', 'outputs', 'predictions', file_name.replace('_H264', '') + '.json')

    with open(json_file_path) as f:
        pose_sequence = json.load(f)

    person_id = st.sidebar.selectbox('Choose a person', [i for i in range(len(pose_sequence[0]['instances']))])
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Number of persons detected on the first frame: ", len(pose_sequence[0]['instances']))
    with col2:
        st.write("Number of frames: ", len(pose_sequence))

    visulization_path = os.path.join(folder_path, selected_video_file)
    st.session_state['visulization_path'] = visulization_path

    cap = cv2.VideoCapture(visulization_path)
    ret, frame = cap.read() 
    # convert frame to RGB format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for instance_id in range(len(pose_sequence[0]['instances'])):
            draw_bounding_box_and_label(frame, 
                                        pose_sequence[0]['instances'][instance_id]['bbox'][0], 
                                        pose_sequence[0]['instances'][instance_id]['bbox_score'], 
                                        "Person " + str(instance_id), 
                                        colors[instance_id % len(colors)])
    col1, col2 = st.columns(2)
    with col1:
        st.image(frame)
    cap.release()
    with col2:
            for instance_id in range(len(pose_sequence[0]['instances'])):
                st.write("Person " + str(instance_id) + ": " + color_names[instance_id % len(colors)] + ". Probability: " + str(100*round(pose_sequence[0]['instances'][instance_id]['bbox_score'], 3)) + "%")


    st.markdown('## General information and type of movement')

    movement = st.selectbox('Choose a type of movement', ['Squat', 'Deadlift', 'Bench press'])
    st.session_state['movement'] = movement

    if movement == 'Squat':
        squat_analyzer.main(json_file_path, person_id)

    else:
        st.write("Number of persons detected on the image: ", len(pose_sequence))
        st.image(os.path.join('UserInterface', 'outputs', 'visualizations', file_name + '.jpg'))
