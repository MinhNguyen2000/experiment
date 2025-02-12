# TODO - Import the important libraries
import cv2
cam = cv2.VideoCapture('/dev/video0')

if not cam.isOPened():
    print("Error: Could not open camera")
    exit()

# TODO - YOLO detection
def yolo_detection(image, yolo_model):
    '''
    This function implements YOLO object detectio'''
    # Preprocess the image for YOLO detection
    
    # Predict the bounding boxes and clas probabilities

    # Return the prediction
    pass

# TODO - R-CNN detection
# TODO - SSD detection

def display_detection():
    '''This helper function is for displaying the detection results'''

while True:
    # Obtain the camera frames from the camera sensor
    ret, frame = cam.read()
    if not ret:
        print("Could not read frame")
        break

    # Preprocess the frame
    

    # Show the image frame with overlays 
    cv2.imshow('Camera frame', frame)

    # Exit the loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break