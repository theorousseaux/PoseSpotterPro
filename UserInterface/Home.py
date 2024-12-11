import streamlit as st
import sys
import os
sys.path.append(os.getcwd())
from UserInterface.utils import add_logo


add_logo("UserInterface/HEADMIND PARTNERS AI -  Bleu.png", height=300)

st.title("Home")
st.subheader("How it works ")
#st.write("This is a web application that allows you to upload files with human instances and get the predictions of the model.")
st.write("This web application allows you to apply pose estimation models to uploaded files or to your webcam feed. The application is divided into 6 tabs:")
st.write('- **:man_dancing: Inference tab** : this tab allow you to choose the model you want to use and upload the file you want to predict.')
st.write('- **:camera: Webcam Record tab** : this tab allow you to apply a pose estimation model on your webcam feed and see the predictions in real time.')
st.write('- **:file_folder: Outputs tab** : this tab display all the inference results.')
st.write('- **:woman-cartwheeling: Movement Analyzer tab** : this tab allow you to choose a video file and apply an analysis to check the correctness of the movement.')
st.write('- **:leaves: Fall Detection tab** : this tab allow you to choose a video file and apply an analysis to check if a fall has occurred.')
