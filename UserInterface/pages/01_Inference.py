import streamlit as st
import os
import sys
import time
sys.path.append(os.getcwd())
from PoseEstimation.pose_inferencer import PoseInferencer, det_dict, pose_dict
from UserInterface.utils import add_logo, save_uploaded_file
from MediaLoader.utils import get_video_info_dict, encode_video_H264


add_logo("UserInterface/HEADMIND PARTNERS AI -  Bleu.png", height=300)


################## Functions ##################

def load_inferencer(detector, pose_estimator):
    st.session_state['inferencer'] = PoseInferencer(detector=detector, pose_estimator=pose_estimator)

def streamlit_callback(num_frames, fps, info_dict):
    fps = round(fps, 2)
    time_left = round((info_dict['num_frames'] - num_frames) / fps, 2)
    text = "Inference in progress, frame {}/{}, {} FPS ({} seconds left)".format(num_frames, info_dict['num_frames'], fps, time_left)
    progress_bar.progress(num_frames/info_dict['num_frames'], text=text)

################## Session state ##############

# Store the inferencer in the session state
if 'inferencer' not in st.session_state:
    st.session_state['inferencer'] = None

# Store the file path in the session state, this path is used with the 'Inference' page only
if 'file_path' not in st.session_state:
    st.session_state['file_path'] = None

################## Sidebar ##################

detector = st.sidebar.selectbox("Select a detector", det_dict.keys())
pose_estimator = st.sidebar.selectbox("Select a pose estimator", pose_dict.keys())


models_button = st.sidebar.button("Select models", on_click=load_inferencer, args=(detector, pose_estimator))

################## Main ##################

st.title(":man_dancing: Inference")

if st.session_state['inferencer'] is not None:
    st.write("Pose estimator loaded : " + st.session_state['inferencer'].pose_estimator_name)
    st.write("Detector loaded : " + st.session_state['inferencer'].detector_name)
else:
    st.write(':heavy_exclamation_mark: **WARNING** : You need to load a *Pose Estimation model* first :heavy_exclamation_mark:')

st.write('## :tv: Upload a video')
file = st.file_uploader("Upload a file", type=["mp4", "jpg", "png", ".mov", ".avi"])

if file is not None:
    st.session_state['file_path'] = save_uploaded_file(file)

if file is not None and st.session_state['inferencer'] is not None:
    submit_button = st.button("Submit")

    if submit_button:

        # if the file is an image, we run the inference on the image
        if st.session_state['file_path'].endswith(("jpg", "png")):
            st.session_state['inferencer'].process_one_image(st.session_state['file_path'], output_directory=os.path.join('UserInterface', 'outputs') + os.sep)
            st.subheader("Visualisation of the result")
            st.image(os.path.join('UserInterface', 'outputs', 'visualizations', st.session_state['file_path'].split(os.sep)[-1]), use_column_width=True)

        elif st.session_state['file_path'].endswith((".mp4", ".mov", ".avi", ".MOV", ".MP4", ".AVI")):
            progress_bar = st.progress(0, text="Inference in progress")
            st.session_state['inferencer'].process_one_video(st.session_state['file_path'], output_directory=os.path.join('UserInterface', 'outputs') + os.sep, callback=streamlit_callback)

            st.subheader("Visualisation of the result")
            file_name = st.session_state['file_path'].split(os.sep)[-1]
            visualization_path = encode_video_H264(os.path.join('UserInterface', 'outputs', 'visualizations', file_name), remove_original=True)
            st.video(visualization_path)


        st.write("JSON file saved in `UserInterface/outputs/predictions/`")
        st.write("Image saved in `UserInterface/outputs/visualizations/`")

print(st.session_state)
