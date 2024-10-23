import cv2
import numpy as np
from utils import load_model_from_path, preprocess_image, detect_basic_shapes

# Load pre-trained model
model_path = 'models/shape_model.h5'
model = load_model_from_path(model_path)

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect basic shapes using OpenCV
    shape_name = detect_basic_shapes(frame)
    
    # Preprocess the frame for advanced shape recognition
    processed_frame = preprocess_image(frame)
    
    # Use the model to predict advanced shapes
    predictions = model.predict(processed_frame)
    advanced_shape_name = np.argmax(predictions)

    # Display the detected shape
    cv2.putText(frame, f'Basic Shape: {shape_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Advanced Shape: {advanced_shape_name}', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Shape Detection', frame)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()