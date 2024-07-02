from ultralytics import YOLO
import cv2
import math

def prediction(path_x):
    prediction = ''
    video_capture = path_x
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    
    model=YOLO("YOLO-Weights/yolov8-24.pt")
    classNames = ['Anthracnose', 'Healthy', 'Mealy bugs']

    while True:
        success, img = cap.read()
        if not success:
            break  # Exit the loop if there are no more frames
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            num_predictions = len(boxes)
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                return class_name, conf




def video_detection(path_x):
    video_capture = path_x
    
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    
    model=YOLO("YOLO-Weights/yolov8-24.pt")
    classNames = ['anthracnose', 'healthy', 'mealy bugs']

    
    while True:
        success, img = cap.read()
        if not success:
            break  # Exit the loop if there are no more frames
        results=model(img,stream=True)
        for r in results:
            boxes=r.boxes
            num_predictions = len(boxes)
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                prediction = label

                print ("Prediction", label)
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=6)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                if class_name == 'anthracnose':
                    color=(0, 204, 255)
                elif class_name == "mealy bugs":
                    color = (222, 82, 175)
                elif class_name == "healthy":
                    color = (50, 205, 50)
                if conf>0.35 or (num_predictions==1 and conf>=0.15):
                    cv2.rectangle(img, (x1,y1), (x2,y2), color,30)  
                    cv2.rectangle(img, (x1,y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255],lineType=cv2.LINE_AA)
            yield img

        
cv2.destroyAllWindows()

        
        