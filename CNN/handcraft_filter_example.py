import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale for processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Sobel filters (independent)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Vertical edges
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Horizontal edges
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)     # Combined magnitude
    
    # Apply independent vertical edge filter
    vertical_kernel = np.array([[-1, 0, 1],
                               [-2, 0, 2],
                               [-1, 0, 1]], dtype=np.float32)
    vertical_edges = cv2.filter2D(gray, cv2.CV_64F, vertical_kernel)
    
    # Normalize all outputs to 0-255
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    vertical_edges = cv2.normalize(vertical_edges, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Display all three windows
    cv2.imshow('Original Image', frame)
    cv2.imshow('Sobel Filter (Combined Edges)', sobel_combined)
    cv2.imshow('Vertical Edge Filter', vertical_edges)
    
    # Exit on ESC key
    if cv2.waitKey(1) == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()