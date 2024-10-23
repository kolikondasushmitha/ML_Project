import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Define the path to the dataset folder
DATASET_PATH = 'datasets/shapes/'

# Label mapping for basic shapes and advanced shapes
LABELS = {
    'circle': 0,
    'square': 1,
    'triangle': 2,
    'advanced_shapes': 3
}

# Function to load the dataset
def load_shape_data(image_size=(64, 64)):
    images = []
    labels = []
    
    # Iterate through each shape folder (circle, square, etc.)
    for shape_name, label in LABELS.items():
        shape_folder = os.path.join(DATASET_PATH, shape_name)
        
        # Iterate through all images in the shape folder
        for image_name in os.listdir(shape_folder):
            image_path = os.path.join(shape_folder, image_name)
            image = cv2.imread(image_path)
            
            if image is not None:
                resized_image = cv2.resize(image, image_size)
                normalized_image = resized_image / 255.0
                
                images.append(normalized_image)
                labels.append(label)
    
    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)
    
    # Split the data into training and testing sets (80% train, 20% test)
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    return train_images, train_labels, test_images, test_labels

# Function to preprocess an image for model prediction
def preprocess_image(image):
    resized_image = cv2.resize(image, (64, 64))
    normalized_image = resized_image / 255.0
    return np.expand_dims(normalized_image, axis=0)

# Function to detect basic shapes using contours
def detect_basic_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) == 3:
            return 'Triangle'
        elif len(approx) == 4:
            return 'Square'
        elif len(approx) > 4:
            return 'Circle'
    
    return 'Unknown'

# Function to load a pre-trained model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

