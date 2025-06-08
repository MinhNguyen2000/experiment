from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
import numpy as np

# Load the saved model 
facetracker = load_model('models/facetracker.h5', compile=False)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,-6)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("There are some errors in the capture process")
        break

    # Ensure that the camera captures a 450:450 frame
    frame = frame[50:500, 50:500, :]

    rgb =cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # cv2.imshow("Face Detection", frame)
    resized = tf.image.resize(rgb,(120,120))
    # cv2.imshow("Face Detection", resized)

    yhat = facetracker.predict(np.expand_dims(resized/255,0))
    sample_coords = yhat[1][0]

    detection_threshold = 0.3
    if yhat[0] > detection_threshold:
        x1, y1, x2, y2 = np.multiply(sample_coords, [450, 450, 450 ,450]).astype(int)
        print(x1, y1, x2, y2)

        # cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        # cv2.rectangle(frame, (x1,y1-20), (x1+80,y1), (255,0,0), -1)
        # cv2.putText(frame, 'face',(x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        cv2.rectangle(frame,
                      tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                      tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                      (255,0,0), 2)
    
    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()