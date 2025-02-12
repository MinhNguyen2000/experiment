import os 
import time
import uuid     # To create nique uniform identifier for collected images
import cv2      # For access camera sensors for image data collection

IMAGES_PATH = os.path.join('data','images')
number_images = 5

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 960)

# for imgnum in range(number_images):
#     print(f'Collecting image {imgnum}')
#     ret, frame = cap.read()

#     if not ret:
#         print("There are some errors in the capture process")
#         break
#     else:
#         imgname = os.path.join(IMAGES_PATH,f'{str(uuid.uuid1())}.jpg')
#         cv2.imwrite(imgname, frame)
#         cv2.imshow('Frame', frame)
#         time.sleep(0.5)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error")
        break
    else:
        
        # time.sleep(0.5)
        print("Hello")
        cv2.imshow("Video stream",frame)

    if cv2.waitKey(1) == ord('q'):
        print("User quit")
        break

    

cap.release()
cv2.destroyAllWindows()