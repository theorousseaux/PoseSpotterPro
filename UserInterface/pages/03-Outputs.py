import streamlit as st
import os
import sys
from UserInterface.utils import add_logo, prune_outputs
sys.path.append(os.getcwd())


add_logo("UserInterface/HEADMIND PARTNERS AI -  Bleu.png", height=300)

st.title(":file_folder: Outputs")

col1, col2 = st.columns(2)

with col1:
    st.write("This page displays all the inference results")
with col2:
    delete_button = st.button("Delete all outputs", on_click=prune_outputs)

# Spécifiez le répertoire ici
directory = 'UserInterface/outputs/visualizations/'

# Liste toutes les extensions de fichier autorisées
image_extensions = ['.jpg', '.png', '.jpeg', '.bmp']
video_extensions = ['.mp4']

# Liste tous les fichiers dans le répertoire
files = os.listdir(directory)

# Trie les fichiers par leur nom
files.sort()

# Compteur pour suivre le nombre d'images affichées
counter = 0

# Parcourir tous les fichiers
for file in files:
    # Obtient l'extension du fichier
    extension = os.path.splitext(file)[1]

    # Si c'est une image, utilise st.image
    if extension.lower() in image_extensions:
        # Si le compteur est divisible par 3, crée une nouvelle ligne de colonnes
        if counter % 2 == 0:
            cols = st.columns(2)

        # Affiche l'image dans la colonne actuelle
        cols[counter % 2].image(os.path.join(directory, file), use_column_width=True)

        # Incrémente le compteur
        counter += 1

    # Si c'est une vidéo, utilise st.video
    elif extension.lower() in video_extensions:
       
        if counter % 2 == 0:
            cols = st.columns(2)
        
        # Affiche la vidéo dans la colonne actuelle
        with cols[counter % 2]:
            st.video(os.path.join(directory, file))

        # Incrémente le compteur
        counter += 1

