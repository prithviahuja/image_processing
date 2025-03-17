from ultralytics import YOLO
import cv2
import cvzone
import math

cap=cv2.VideoCapture(0)    ## sset the video camera as cap
cap.set(3,1280)
cap.set(4,720)    # set dimention of the camera screen
model=YOLO("yolov8n.pt")
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]
#predifined coco daatset on which yolo is trained

while True:       ## use while loop(inifinite loop) until the window is closed to capture continuoes images
    success,img=cap.read()   #success is booloean true/false to check if the image is successfully captured
    results=model(img,stream=True) #process the frames being captured using yolo
    for r in results:
        boxes=r.boxes  #extracts all detectes objects
        for box in boxes:

            #bounding box
            x1,y1,x2,y2=box.xyxy[0]  #gives coordinates of the detected objects 
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)  #store the coordinates as integer as cv requires integer values
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,200,0),3) #Set values,clour,thickness of boxes 

            #confidence
            conf=math.ceil((box.conf[0]*100))/100  #used to confidence score and round the values to 2 digit decimal
              #put the confidence level and its position

            #class name
            cls=int(box.cls[0])   #give integger value for each object being detected
            
            cvzone.putTextRect(img,f'{classNames[cls]}{conf}',(x1,y1-20),scale=1,thickness=1)#put the confidence level along with classname from given dataset and its position,scale,thickness

    cv2.imshow("image",img)  #give the name of the window as image
    cv2.waitKey(1)        #milliseconds to wait till a key is pressed
                        #if we keep it 0 it will go to next frame only when we press a key

