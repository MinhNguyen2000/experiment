import cv2
from ultralytics import YOLO

# cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 960)

# Load a pretrained YOLO11n model
model = YOLO("yolo-Weights/yolov8n.pt")
# i=0

# Run inference on the source
while True: 
    # ret,frame = cap.read()
    results = model(source=0, stream=True, show=True, show_labels = True)  # generator of Results objects
    # i += 1
    # print(i)
    # for r in results:
    #     for box in r.boxes:
    #         print(box)

    # cv2.imshow("Webcam",frame)

    if cv2.waitKey(1) == ord('q'):
        break