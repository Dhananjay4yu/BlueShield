import tensorflow as tf
from tensorflow.keras.models import load_model

# Use the 'custom_objects' parameter to load legacy models that used older Keras functions
model = load_model('garbage_classification_model.h5', compile=False)