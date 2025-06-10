import os
import subprocess
import sys
import shutil

# Nom de l'environnement
env_name = "venv38"

# === Étape 1 : Vérifier si Python 3.8 est disponible ===
print("🔍 Vérification de Python 3.8 ...")
try:
    version_output = subprocess.check_output(["py", "-3.8", "--version"], stderr=subprocess.STDOUT)
    print("✅ Python 3.8 est disponible :", version_output.decode().strip())
except subprocess.CalledProcessError:
    print("❌ Python 3.8 n'est pas installé ou non détecté par la commande py -3.8.")
    sys.exit(1)

# === Étape 2 : Créer l'environnement virtuel ===
if os.path.exists(env_name):
    print(f"ℹ Environnement {env_name} existe déjà. Suppression pour recréer proprement...")
    shutil.rmtree(env_name)

print(f"⚙ Création de l'environnement virtuel '{env_name}' avec Python 3.8...")
subprocess.run(["py", "-3.8", "-m", "venv", env_name], check=True)

# === Étape 3 : Activer l'environnement ===
activate_path = os.path.join(env_name, "Scripts", "activate")
print(f"✅ Environnement créé. Pour l'activer manuellement, exécutez dans PowerShell :\n\n    .\\{activate_path}\n")

# === Étape 4 : Installer les dépendances ===
pip_path = os.path.join(env_name, "Scripts", "pip.exe")
print("📥 Installation des dépendances...")
subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
print("✅ Dépendances installées avec succès !")

import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO

# === Charger le modèle ===
MODEL_URL = "https://drive.google.com/file/d/1D0zSzVpd31I_Gcz5_FYkUG9FkkK87ktx/view?usp=drive_link"


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Téléchargement du modèle..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(r.content)
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === Fonction de prédiction ===
def predict(image):
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = "Chien 🐶" if prediction > 0.5 else "Chat 🐱"
    return label, confidence * 100

# === Interface utilisateur ===
st.set_page_config(page_title="Classificateur Chat vs Chien 🐾", layout="centered")
st.title("🐾 Classificateur Chat vs Chien")

# Onglets pour différentes options
tab1, tab2 = st.tabs(["📁 Image locale", "🌐 Image via URL"])

with tab1:
    uploaded_file = st.file_uploader("Choisis une image (JPG, PNG)...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Image chargée", use_column_width=True)

        if st.button("Analyser l’image locale"):
            with st.spinner("Analyse en cours..."):
                label, confidence = predict(image)
            st.success(f"Résultat : *{label}* ({confidence:.2f} % de confiance)")

with tab2:
    url = st.text_input("Entrez l'URL de l'image")
    if st.button("Analyser l’image depuis l’URL") and url:
        try:
            response = requests.get(url)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(image, caption="Image depuis l’URL", use_column_width=True)

            with st.spinner("Analyse en cours..."):
                label, confidence = predict(image)
            st.success(f"Résultat : *{label}* ({confidence:.2f} % de confiance)")
        except Exception as e:
            st.error(f"Erreur : Impossible de charger l'image. Détail : {e}")
