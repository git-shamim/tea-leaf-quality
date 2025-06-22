import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from utils import preprocess_image

# Model paths
MOUNTED_MODEL_PATH = "models/tea_model.h5"
MODEL_LOCAL_PATH = "models/tea_model.h5"

@st.cache_resource
def load_model():
    try:
        if os.path.exists(MOUNTED_MODEL_PATH):
            st.info("📦 Loading model from mounted volume...")
            return tf.keras.models.load_model(MOUNTED_MODEL_PATH)

        elif os.path.exists(MODEL_LOCAL_PATH):
            st.info("💾 Loading model from local path...")
            return tf.keras.models.load_model(MODEL_LOCAL_PATH)

        else:
            st.error("❌ No model found at either mounted volume or local path.")
            st.stop()

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

model = load_model()
if model is None:
    st.stop()

class_names = [
    "T1 - Fresh (1–2 days)",
    "T2 - Mid-aged (3–4 days)",
    "T3 - Aged (5–7 days)",
    "T4 - Old (7+ days)"
]

class_explanations = {
    "T1 - Fresh (1–2 days)": "🍃 Top quality: Ideal for premium tea.",
    "T2 - Mid-aged (3–4 days)": "✅ Good quality: Commercial blends.",
    "T3 - Aged (5–7 days)": "⚠️ Moderate: Less flavor, lower-grade.",
    "T4 - Old (7+ days)": "❌ Not recommended: Overaged for tea."
}

st.sidebar.title("🫖 Tea Quality Classifier")
st.sidebar.markdown("""
This app classifies tea leaves into T1–T4 grades using a deep learning model trained on the [TeaLeafAgeQuality Dataset](https://data.mendeley.com/datasets/7t964jmmy3/1).
""")

st.title("🍵 Predict Tea Leaf Quality from Image")

uploaded_file = st.file_uploader("Upload a tea leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image")

        img_array = preprocess_image(uploaded_file)

        prediction = model.predict(img_array)
        confidence = np.max(prediction)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_names[predicted_class_index]

        st.subheader("🔍 Prediction")
        if confidence < 0.6:
            st.warning(f"⚠️ Low confidence: {confidence:.2}")
        else:
            st.success(f"✅ Predicted Class: {predicted_class}")
            st.info(f"💡 {class_explanations[predicted_class]}")

        st.markdown(f"**Confidence:** {confidence:.2%}")

    except Exception as e:
        st.error(f"❌ Could not process the image. Error: {e}")
else:
    st.info("Please upload a tea leaf image to begin.")
