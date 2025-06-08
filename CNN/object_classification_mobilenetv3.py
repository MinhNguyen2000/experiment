import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Load MobileNetV3 model with ImageNet weights
model = MobileNetV3Small(weights='imagenet',include_top=True)

# Display parameters
font_scale = 0.7
thickness = 2
color = (0, 0, 255)  # Green
font = cv2.FONT_HERSHEY_SIMPLEX
y_start = 30  # Starting Y position for labels
padding = 5    # Space between labels

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Preprocess frame
    resized = cv2.resize(frame, (224, 224))
    x = preprocess_input(resized)
    x = np.expand_dims(x, axis=0)

    # Get predictions
    preds = model.predict(x)
    decoded_preds = decode_predictions(preds, top=3)[0]

    # Calculate maximum text width and total height needed
    max_width = 0
    total_height = 0
    labels = []
    
    for _, class_name, confidence in decoded_preds:
        label = f"{class_name}: {confidence*100:.1f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )
        max_width = max(max_width, text_width)
        total_height += text_height + baseline + padding
        labels.append((label, text_height, baseline))

    # Draw unified background rectangle
    if labels:
        cv2.rectangle(frame,
            (10, y_start - total_height),
            (10 + max_width, y_start + labels[-1][2]),
            (0, 0, 0), -1)

    # Draw labels with proper spacing
    current_y = y_start
    for label, text_height, baseline in reversed(labels):
        current_y -= (text_height + baseline + padding)
        cv2.putText(frame, label, (10, current_y + text_height), 
                   font, font_scale, color, thickness)

    # Display frame
    cv2.imshow('MobileNetV3 Classification', frame)

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()