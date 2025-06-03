import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained CNN model
model = load_model('best_model.h5')  # Update with the path to your model

# Define the list of class names corresponding to your model's output classes
class_names_list = ['Cardboard', 'Bottle', 'Can']  # Replace with your actual class names

def preprocess_image(image):
    """Preprocess the image for the model."""
    input_size = (model.input_shape[1], model.input_shape[2])  # Get input size from the model
    resized_image = cv2.resize(image, input_size)  # Resize
    normalized_image = resized_image / 255.0  # Normalize
    input_data = np.expand_dims(normalized_image, axis=0)  # Expand dimensions
    return input_data

def detect_garbage(image):
    """Run garbage detection on the image."""
    input_data = preprocess_image(image)

    # Predict using the CNN model
    predictions = model.predict(input_data)

    # Assuming the model outputs bounding boxes and class probabilities
    # For this example, let's assume we generate dummy bounding boxes
    # In practice, you would get these from your model's output
    boxes = []  # List to hold the bounding boxes
    scores = predictions[0]  # Confidence scores (assuming model returns these)

    # Simulating bounding box coordinates and class indices
    for i in range(len(scores)):
        if scores[i] > 0.5:  # Confidence threshold
            # Dummy box coordinates (replace with actual bounding box coordinates)
            x, y, width, height = (50 + i * 30, 50 + i * 20, 100, 100)
            boxes.append((x, y, width, height, scores[i]))

    return boxes, image

# Streamlit UI
st.title("Garbage Detection Application")
st.write("Upload an image to detect garbage.")

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the uploaded image file
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert to numpy array for OpenCV processing

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform garbage detection
    boxes, result_image = detect_garbage(image)

    # Draw bounding boxes on the original image
    for box in boxes:
        x, y, width, height, box_confidence = box
        cv2.rectangle(result_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        # Use the predicted class from your class_names_list (using dummy class index)
        predicted_class = class_names_list[0]  # Update this logic to match your detection
        cv2.putText(result_image, f'{predicted_class}: {box_confidence:.2f}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the result image back to RGB for displaying in Streamlit
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    st.image(result_image_rgb, caption="Detection Result", use_column_width=True)