import streamlit as st
import numpy as np
import pandas as pd
import os
import cv2
import tempfile
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

IMG_SIZE = 224
SEQ_LENGTH = 20
NUM_FEATURES = 2048


# -----------------------------
# FACE DETECTION
# -----------------------------
def crop_face_center(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return frame[y:y+h, x:x+w]

    # fallback center crop
    h, w, _ = frame.shape
    size = min(h, w)
    y1 = (h - size) // 2
    x1 = (w - size) // 2
    return frame[y1:y1+size, x1:x1+size]


# -----------------------------
# LOAD VIDEO
# -----------------------------
def load_video(uploaded_file, resize=(IMG_SIZE, IMG_SIZE)):
    temp_file_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    cap = cv2.VideoCapture(temp_file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(total_frames / SEQ_LENGTH), 1)

    frames = []

    try:
        for frame_cntr in range(SEQ_LENGTH):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cntr * skip_frames_window)
            ret, frame = cap.read()
            if not ret:
                break

            frame = crop_face_center(frame)
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    finally:
        cap.release()
        os.remove(temp_file_path)

    return np.array(frames)


# -----------------------------
# FEATURE EXTRACTOR
# -----------------------------
def build_feature_extractor():
    from tensorflow.keras.applications import InceptionV3

    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    preprocess_input = tf.keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = preprocess_input(inputs)
    outputs = base_model(x)

    return keras.Model(inputs, outputs)


# -----------------------------
# GRADCAM MODEL (FIXED)
# -----------------------------
def build_gradcam_model():
    base_model = tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    last_conv_layer = base_model.get_layer("mixed10")

    grad_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[last_conv_layer.output, base_model.output],
    )

    return grad_model


# Load models
sequence_model = load_model("models/inceptionNet_model.h5")
feature_extractor = build_feature_extractor()
grad_model = build_gradcam_model()


# -----------------------------
# PREPARE VIDEO
# -----------------------------
def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros((1, SEQ_LENGTH), dtype="bool")
    frame_features = np.zeros((1, SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(SEQ_LENGTH, video_length)

        for j in range(length):
            features = feature_extractor.predict(
                batch[None, j, :], verbose=0
            )
            frame_features[i, j, :] = features

        frame_mask[i, :length] = 1

    return frame_features, frame_mask


# -----------------------------
# FRAME IMPORTANCE
# -----------------------------
def compute_frame_importance(frame_features, frame_mask):
    features_var = tf.Variable(frame_features, dtype=tf.float32)

    with tf.GradientTape() as tape:
        pred = sequence_model([features_var, tf.constant(frame_mask)])
        fake_score = pred[0, 0]

    grads = tape.gradient(fake_score, features_var)

    n_frames = int(frame_mask[0].sum())
    importance = tf.norm(grads[0], axis=-1).numpy()[:n_frames]

    if importance.max() > 0:
        importance = importance / importance.max()

    shap_values = tf.reduce_sum(
        grads[0, :n_frames, :] * features_var[0, :n_frames, :], axis=-1
    ).numpy()

    return importance, grads, shap_values


# -----------------------------
# GRADCAM
# -----------------------------
def generate_gradcam(frame, feat_weight):
    img = tf.cast(frame[np.newaxis, ...], tf.float32)
    img = tf.keras.applications.inception_v3.preprocess_input(img)

    with tf.GradientTape() as tape:
        conv_out, predictions = grad_model(img)
        target = tf.reduce_sum(predictions * feat_weight)

    grads = tape.gradient(target, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_out, axis=-1)

    heatmap = tf.nn.relu(heatmap)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    heatmap = heatmap.numpy()
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))

    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlay


# -----------------------------
# EXPLAIN
# -----------------------------
def explain_prediction(frames, frame_features, frame_mask):
    importance, grads, shap_values = compute_frame_importance(
        frame_features, frame_mask
    )

    top_indices = np.argsort(importance)[-3:][::-1]
    gradcam_results = []

    for idx in top_indices:
        if idx >= len(frames):
            continue

        feat_weight = grads[0, idx]
        overlay = generate_gradcam(frames[idx], feat_weight)

        gradcam_results.append(
            (int(idx), overlay, float(importance[idx]))
        )

    return importance, shap_values, gradcam_results


# -----------------------------
# PREDICTION
# -----------------------------
def predict_video(video_file):
    frames = load_video(video_file)
    frame_features, frame_mask = prepare_single_video(frames)

    probabilities = sequence_model.predict(
        [frame_features, frame_mask], verbose=0
    )[0]

    fake_probability = probabilities[0] * 100
    return fake_probability, frames, frame_features, frame_mask


# -----------------------------
# STREAMLIT UI
# -----------------------------
def main():
    st.title("DeepFake Detection")
    st.image("Deepfake.png")
    st.write("Upload a video to predict if it's fake or real.")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])

    if uploaded_file is not None:
        st.video(uploaded_file)

        if st.button("Detect"):
            fake_probability, frames, frame_features, frame_mask = predict_video(uploaded_file)

            result = "Fake" if fake_probability > 50 else "Real"
            st.subheader(f"Prediction: {result} ({fake_probability:.1f}%)")

            importance, shap_values, gradcam_results = explain_prediction(
                frames, frame_features, frame_mask
            )

            frame_labels = [f"Frame {i+1}" for i in range(len(importance))]

            st.subheader("SHAP Values")
            shap_df = pd.DataFrame({"SHAP Value": shap_values}, index=frame_labels)
            st.bar_chart(shap_df)

            st.subheader("Frame Importance")
            imp_df = pd.DataFrame({"Importance": importance}, index=frame_labels)
            st.bar_chart(imp_df)

            st.subheader("Grad-CAM Results")
            for idx, overlay, imp in gradcam_results:
                st.image(
                    overlay,
                    caption=f"Frame {idx+1} | Importance: {imp:.3f}",
                )


if __name__ == "__main__":
    main()