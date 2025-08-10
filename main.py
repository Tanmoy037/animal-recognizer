import os
import ssl
import certifi
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions

# Fix SSL cert verification on macOS when downloading pretrained weights
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

# Load pre-trained InceptionV3
model = InceptionV3(weights='imagenet')

# Load and preprocess your image
img_path = '/Users/tanmoy/development/image-recognition/images (4).jpeg'
img = image.load_img(img_path, target_size=(299, 299))  # InceptionV3 needs 299x299
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make prediction
predictions = model.predict(img_array)

# Decode predictions to get animal names
decoded = decode_predictions(predictions, top=3)[0]

# Print results
for i, (imagenet_id, label, score) in enumerate(decoded):
    print(f"{i+1}: {label} ({score:.2f})")