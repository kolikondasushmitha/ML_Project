import cv2
import numpy as np
import tensorflow as tf  
from ML_Project.utils import preprocess_frame, load_model, identify_shape

def main():
    # Load the trained model for advanced shapes
    model = load_model('models/shape_model.h5')  # For TensorFlow
    # model = torch.load('models/shape_model.pth')  # For PyTorch

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        processed_frame = preprocess_frame(frame)

        # Detect contours for basic shapes
        gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
            x = approx.ravel()[0]
            y = approx.ravel()[1] - 10

            if len(approx) == 3:
                cv2.putText(frame, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif len(approx) == 4:
                cv2.putText(frame, "Square", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            elif len(approx) > 4:
                cv2.putText(frame, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Use the trained ML model for advanced shape detection
        prediction = model.predict(processed_frame)  # TensorFlow
        # prediction = model(processed_frame)  # PyTorch
        
        shape_name = identify_shape(prediction)
        cv2.putText(frame, shape_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show the result
        cv2.imshow('Shape Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
