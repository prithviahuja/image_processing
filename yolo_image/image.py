from ultralytics import YOLO
import cv2
model= YOLO('yolov8l.pt')# use the type of model (yolov8m.pt/yolov8n.pt etc)
results=model("C:\Python\image_detection\image\WhatsApp Image 2025-03-17 at 23.51.21.jpeg",show=True)
cv2.waitKey(0)#milliseconds to wait till a key is pressed
