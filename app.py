"""
Pneumonia Detection from Chest X-Rays — Streamlit App
Model: VGG16 (fine-tuned) trained via transfer learning, with Grad-CAM explainability.

Deployment notes:
- Model file must be available at MODEL_PATH (default: model.keras) OR downloaded
  at runtime from Google Drive using the GDRIVE_FILE_ID below.
- Class indices from training: {'NORMAL': 0, 'PNEUMONIA': 1}
- Input size: 224x224, rescaled to [0,1]
"""

import os
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import Model
import cv2
from PIL import Image

# ----------------------------- CONFIG -----------------------------
IMG_SIZE = (224, 224)
MODEL_PATH = "model.keras"  # place VGG16_finetuned_final.keras here, renamed to this

# OPTIONAL: if the model file isn't committed to the repo (too large for GitHub),
# set this to your Google Drive file ID and the app will download it on first run.
# Get the ID from your shareable link: https://drive.google.com/file/d/<THIS_PART>/view
GDRIVE_FILE_ID = ""  # e.g. "1AbCdEfGhIjKlMnOpQrStUvWxYz"

CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}

st.set_page_config(page_title="Pneumonia Detector (Grad-CAM)", page_icon="🫁", layout="centered")


# ------------------------- MODEL LOADING ---------------------------
def download_model_if_needed():
    if os.path.exists(MODEL_PATH):
        return
    if not GDRIVE_FILE_ID:
        st.error(
            f"Model file '{MODEL_PATH}' not found and no GDRIVE_FILE_ID is set. "
            "Either commit the model file to the repo or set GDRIVE_FILE_ID in app.py."
        )
        st.stop()
    with st.spinner("Downloading model from Google Drive (first run only)..."):
        import gdown
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    download_model_if_needed()
    return tf.keras.models.load_model(MODEL_PATH)


# ------------------------- GRAD-CAM LOGIC ---------------------------
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        try:
            shape = layer.output.shape
        except AttributeError:
            continue
        if shape is not None and len(shape) == 4:
            return layer.name
    raise ValueError("No 4D conv layer found in this model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, 0]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_gradcam(pil_img, heatmap, alpha=0.4):
    img = np.array(pil_img.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE)
    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlayed


def preprocess_image(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)


# ------------------------------- UI ---------------------------------
st.title("🫁 Pneumonia Detection from Chest X-Rays")
st.caption("Transfer Learning (VGG16, fine-tuned) + Grad-CAM Explainability")

st.markdown(
    "Upload a chest X-ray image (JPEG/PNG). The model predicts **NORMAL** vs "
    "**PNEUMONIA**, and Grad-CAM highlights the regions the model focused on."
)

uploaded_file = st.file_uploader("Upload chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    pil_img = Image.open(uploaded_file)

    with st.spinner("Running inference..."):
        model = load_model()
        img_array = preprocess_image(pil_img)
        prob = float(model.predict(img_array, verbose=0)[0][0])
        pred_class = 1 if prob >= 0.5 else 0
        label = CLASS_NAMES[pred_class]
        confidence = prob if pred_class == 1 else (1 - prob)

        last_conv = get_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(img_array, model, last_conv)
        overlayed = overlay_gradcam(pil_img, heatmap)

    col1, col2 = st.columns(2)
    with col1:
        st.image(pil_img, caption="Uploaded X-ray", use_container_width=True)
    with col2:
        st.image(overlayed, caption="Grad-CAM Overlay", use_container_width=True)

    st.divider()

    if label == "PNEUMONIA":
        st.error(f"### Prediction: {label}")
    else:
        st.success(f"### Prediction: {label}")

    st.metric("Confidence", f"{confidence * 100:.2f}%")
    st.progress(min(max(confidence, 0.0), 1.0))

    st.warning(
        "This is a research/thesis demo tool, not a certified diagnostic device. "
        "Do not use for real medical decisions — always consult a radiologist/doctor."
    )
else:
    st.info("Upload an X-ray image to get a prediction.")

st.divider()
st.caption("Thesis project: Explainable Deep Learning Framework for Pneumonia Detection "
           "using Transfer Learning and Grad-CAM Visualization.")
