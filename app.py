import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import pickle
import matplotlib.pyplot as plt
from PIL import Image

# Title and description
st.title("🩺 Skin Disease Classifier")
st.write("Upload a skin lesion image and get a prediction using MobileNetV2.")

# Load model and label encoder
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model("skin_disease_classifier.h5", compile=False)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

model, label_encoder = load_model_and_encoder()

# Constants
IMG_SIZE = 224

# Disease descriptions and trusted links
disease_info = {
    "akiec": {
        "name": "Actinic Keratoses and Intraepithelial Carcinoma",
        "description": "A precancerous area of thick, scaly, or crusty skin that often feels dry or rough.",
        "link": "https://www.mayoclinic.org/diseases-conditions/actinic-keratosis"
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "A common type of skin cancer that often appears as a slightly transparent bump.",
        "link": "https://www.webmd.com/cancer/what-is-basal-cell-carcinoma"
    },
    "bkl": {
        "name": "Benign Keratosis",
        "description": "A group of non-cancerous skin growths often caused by sun exposure or aging.",
        "link": "https://dermnetnz.org/topics/seborrhoeic-keratosis"
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "A common benign fibrous skin tumor usually caused by minor skin trauma.",
        "link": "https://www.dermnetnz.org/topics/dermatofibroma"
    },
    "mel": {
        "name": "Melanoma",
        "description": "A dangerous form of skin cancer that arises from pigment-producing melanocytes.",
        "link": "https://www.mayoclinic.org/diseases-conditions/melanoma"
    },
    "nv": {
        "name": "Melanocytic Nevi (Moles)",
        "description": "Common skin growths that are usually benign but should be monitored for changes.",
        "link": "https://www.aad.org/public/diseases/a-z/moles-overview"
    },
    "vasc": {
        "name": "Vascular Lesions",
        "description": "Includes angiomas and hemorrhages – growths involving blood vessels.",
        "link": "https://www.aad.org/public/diseases/a-z/birthmarks"
    }
}

# Preprocessing function
def preprocess_image(image: Image.Image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    st.write("🔍 Classifying...")

    processed_img = preprocess_image(image)
    predictions = model.predict(processed_img)[0]
    predicted_index = np.argmax(predictions)
    predicted_label = label_encoder.inverse_transform([predicted_index])[0]
    confidence = predictions[predicted_index]

    # Display top prediction
    st.success(f"🎯 Predicted: **{predicted_label.upper()}** with **{confidence * 100:.2f}%** confidence.")

    # Description + Link
    info = disease_info.get(predicted_label)
    if info:
        st.markdown(f"ℹ️ **About {info['name']}:** {info['description']}")
        st.markdown(f"[🔗 Learn more from trusted medical source]({info['link']})")

        with st.expander("📘 More Info"):
            st.write(f"**Full Name:** {info['name']}")
            st.write(f"**Detailed Description:** {info['description']}")
            st.write(f"[🔎 Click here to read more on external site]({info['link']})")
    else:
        st.warning("No additional info available for this class.")

    # Probability Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(label_encoder.classes_, predictions, color='lightgreen')
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence Across All Classes")
    ax.set_ylim(0, 1)
    ax.bar_label(bars, fmt="%.2f", padding=3)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)
