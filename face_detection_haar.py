import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Haar classifier - adjust the parameters to optimize performance
    faces = face_cascade.detectMultiScale(frame_gray, 
                                          scaleFactor = 1.1, 
                                          minNeighbors=6, 
                                          minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)
    
    cv2.imshow('Facial Recognition', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    
