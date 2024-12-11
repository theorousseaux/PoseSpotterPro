import streamlit as st
import os
import sys
sys.path.append(os.getcwd())
from Classifier.Fall.fall_inferencer import FallInferencer
from UserInterface.utils import save_uploaded_file
from MediaLoader.utils import encode_video_H264


################## Functions ##################

def streamlit_callback(num_frames, fps, info_dict):
    fps = round(fps, 2)
    progress_bar.progress(num_frames/info_dict['num_frames'], text="Inference in progress, frame {}/{}, {} FPS".format(num_frames, info_dict['num_frames'], fps))

################## Session state ##############

if 'inferencer' not in st.session_state:
    st.session_state['inferencer'] = None

if 'fall_inferencer' not in st.session_state:
    st.session_state['fall_inferencer'] = None

################## Sidebar ##################

upload_video = st.sidebar.selectbox("Upload a video", ["Yes", "No"])

if upload_video == "No":
    folder_path = os.path.join('UserInterface', 'outputs', 'visualizations') + os.sep
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    selected_video_file = st.sidebar.selectbox('Choose a video file', video_files)

################## Main ##################

st.write('# :leaves: Fall Detection')

if st.session_state.inferencer is None:
    st.write(':heavy_exclamation_mark: **WARNING** : You need to load a *Pose Estimation model* first :heavy_exclamation_mark:')
else:
    st.session_state['fall_inferencer'] = FallInferencer(st.session_state.inferencer, window_size=10, step_size=5)

if upload_video == "Yes":
    st.write('## :tv: Upload a video')

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", ".mov", ".avi"])

    if uploaded_file is not None:
        file_path = save_uploaded_file(uploaded_file)
        apply_button = st.button("Apply")

        if apply_button:
            progress_bar = st.progress(0, text="Inference in progress")
            st.session_state.fall_inferencer.process_one_video(file_path, os.path.join('UserInterface', 'outputs', 'fall') + os.sep, callback=streamlit_callback)
            st.subheader("Visualisation of the result")
            file_name = file_path.split(os.sep)[-1]
            visualization_path = encode_video_H264(os.path.join('UserInterface', 'outputs', 'fall', 'visualizations', file_name), remove_original=True)
            st.video(visualization_path)

else:
    if selected_video_file is not None:

        file_name = selected_video_file.split('.')[0]
        json_file_path = os.path.join('UserInterface', 'outputs', 'predictions', file_name.replace('_H264', '') + '.json')

        apply_button = st.button("Apply")
        if apply_button:
            progress_bar = st.progress(0, text="Inference in progress")
            visualization_path = st.session_state.fall_inferencer.process_one_video_without_HPE(video_path=folder_path + selected_video_file, 
                                                                       pred_path=json_file_path,
                                                                       output_directory=os.path.join('UserInterface', 'outputs', 'fall', 'visualizations') + os.sep,
                                                                       callback=streamlit_callback)
            st.subheader("Visualisation of the result")
            visualization_path = encode_video_H264(visualization_path, remove_original=True)
            st.video(visualization_path)

