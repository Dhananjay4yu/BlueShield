import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model and classes
model = tf.keras.models.load_model("trash.h5")
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = (180, 180)

# Streamlit UI
st.set_page_config(page_title="Trash Detector", layout="wide")
st.title("♻️ Real-Time Trash Type Detection")
st.markdown("Show any garbage item to your webcam to identify its type.")

run = st.checkbox('Start Webcam')

frame_placeholder = st.empty()
prediction_text = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access webcam.")
            break

        # Flip horizontally for natural feel
        frame = cv2.flip(frame, 1)

        # Resize & predict
        resized = cv2.resize(frame, IMG_SIZE)
        norm = resized / 255.0
        input_tensor = np.expand_dims(norm, axis=0)

        preds = model.predict(input_tensor)[0]
        class_idx = np.argmax(preds)
        confidence = preds[class_idx]

        label = f"{class_names[class_idx]} ({confidence*100:.2f}%)"

        # Draw label on the frame
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)

        # Convert frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)

        frame_placeholder.image(frame_pil, caption="Live Feed", use_column_width=True)
        prediction_text.markdown(f"### 🧠 Prediction: `{label}`")

    cap.release()
else:
    st.info("Check the box above to start the webcam.")
