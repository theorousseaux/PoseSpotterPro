import base64
import os
import sys
from pathlib import Path
import streamlit as st
sys.path.append(os.getcwd())
import cv2
import numpy as np


logo_url = "UserInterface/HEADMIND PARTNERS AI -  Bleu.png"

# Couleurs courantes
colors = [
    (255, 0, 0),     # Rouge
    (0, 255, 0),     # Vert
    (0, 0, 255),     # Bleu
    (255, 255, 0),   # Jaune
    (0, 255, 255),   # Cyan
    (255, 0, 255),   # Magenta
    (255, 255, 255), # Blanc
    (0, 0, 0),       # Noir
    (128, 0, 0),     # Marron
    (128, 128, 128), # Gris
]

# Noms des couleurs
color_names = [
    "Red",
    "Green",
    "Blue",
    "Yellow",
    "Cyan",
    "Magenta",
    "White",
    "Black",
    "Brown",
    "Grey",
]


def add_logo(logo_url: str, height: int = 120):
    '''Add a logo to the sidebar'''
    logo = f"url(data:image/png;base64,{base64.b64encode(Path(logo_url).read_bytes()).decode()})"
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebar"]{{
                background-image: {logo};
                background-repeat: no-repeat;
                background-position: center bottom 5%;
                background-size: {height}px;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def save_uploaded_file(uploaded_file):
    try:
        file_path = os.path.join('UserInterface', 'Uploaded', uploaded_file.name)
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getvalue())
        return file_path
    except Exception as e:
        st.error(f"Error: {e}")
        return None
    

def prune_outputs():

    # delte the files in the output folder
    output_folder_pred = os.path.join('UserInterface', 'outputs', 'predictions')
    output_folder_vis = os.path.join('UserInterface', 'outputs', 'visualizations')

    for file in os.listdir(output_folder_pred):
        file_path = os.path.join(output_folder_pred, file)
        os.remove(file_path)
    
    for file in os.listdir(output_folder_vis):
        file_path = os.path.join(output_folder_vis, file)
        os.remove(file_path)


def draw_bounding_box_and_label(image: np.array, bbox: list, probability: float, label: str, color: tuple):
    # Convert the bounding box values to integers
    bbox = [int(b) for b in bbox]
    
    # Draw the bounding box on the image
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    # Prepare the text string
    text = f"{label}: {probability*100:.2f}%"
    
    # Define the font scale and thickness
    font_scale = 0.5
    font_thickness = 1

    # Get the text size in pixels
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    
    # Draw a filled rectangle to put the text on top of
    cv2.rectangle(image, (bbox[0], bbox[1] - text_size[1] - 5), (bbox[0] + text_size[0], bbox[1]), color, cv2.FILLED)
    
    # Add the text to the image
    cv2.putText(image, text, (bbox[0], bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)
    
    return image

if __name__ == "__main__":
    prune_outputs() 
