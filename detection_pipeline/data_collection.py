'''
This script is used to collect images from the camera sensors of the device.
'''
import os 
import time
import uuid     # To create nique uniform identifier for collected images
import cv2      # For access camera sensors for image data collection

DIR_PATH = os.getcwd()
IMG_RAW_PATH = os.path.join(DIR_PATH, 'data_raw\\images')
print(IMG_RAW_PATH)
number_images = 10


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_EXPOSURE,-6)

for imgnum in range(number_images):
    print(f'Collecting image {imgnum}')
    
    ret, frame = cap.read()

    if not ret:
        print("There are some errors in the capture process")
        break

    imgname = os.path.join(IMG_RAW_PATH,f'{str(uuid.uuid1())}.jpg')
    cv2.imwrite(imgname, frame)
    cv2.imshow('Frame', frame)
    time.sleep(1.0)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# After collecting the images, you can run labelme from the terminal window to annotate the images