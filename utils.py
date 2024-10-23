import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the pre-trained TensorFlow model
def load_model_from_path(model_path):
    return load_model(model_path)

# Preprocess the input image for the model
def preprocess_image(image, image_size=(64, 64)):
    resized_image = cv2.resize(image, image_size)
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0)

# Load image datasets
def load_shape_data(data_dir):
    labels = []
    images = []
    label_names = os.listdir(data_dir)
    
    for label_name in label_names:
        label_dir = os.path.join(data_dir, label_name)
        for img_file in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_file)
            image = cv2.imread(img_path)
            image = preprocess_image(image)
            images.append(image)
            labels.append(label_name)
    
    return np.array(images), np.array(labels)