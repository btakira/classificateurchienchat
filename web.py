import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import requests
from io import BytesIO

# === Charger le modèle ===
model = tf.keras.models.load_model("chat_vs_chien_model.h5")

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