import cv2
import torch
import numpy as np
import csv
from datetime import datetime
from pathlib import Path
import platform
import socket
import time
import os
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_boxes

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load custom trained YOLOv5 model (fixed loading)
model_path = 'yolov5s.onnx'  # Replace with your model path
model = DetectMultiBackend(model_path, device=device, dnn=False, data=None, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
model.eval()

# Warmup model
imgsz = (640, 640)
if device.type != 'cpu':
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.parameters())))  # warmup

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(1)  # Change to your camera index
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 10)  # Limit FPS for Jetson Nano

# CSV setup
csv_folder = Path('csv')
csv_folder.mkdir(exist_ok=True)
csv_file = csv_folder / 'detections.csv'

# Initialize CSV
if csv_file.exists():
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        last_id = int(rows[-1][0]) if len(rows) > 1 else 0
else:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'person_count', 'see_tv_count', 'see_tv_duration', 'id_device', 'timestamp'])
    last_id = 0

current_id = last_id + 1
see_tv_start_time = {}  # {id: start_time}

# Device info
def get_device_id():
    return socket.gethostname()

id_device = get_device_id()
conf_threshold = 0.3

# FPS counter
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Preprocessing (fixed like detect.py)
    im = letterbox(frame, imgsz[0], stride=stride, auto=pt)[0]  # resize with padding
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0-255 to 0.0-1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    # Inference
    pred = model(im, augment=False, visualize=False)
    
    # NMS (fixed post-processing)
    pred = non_max_suppression(pred, conf_threshold, 0.45, None, False, max_det=1000)

    # Reset counters
    person_count = 0
    see_tv_count = 0
    current_see_tv_ids = set()

    # Process detections
    for det in pred:
        if len(det):
            # Rescale boxes from img_size to frame size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame.shape).round()

            for *xyxy, conf, cls in det:
                if names[int(cls)] == 'person':
                    person_count += 1
                    x1, y1, x2, y2 = map(int, xyxy)

                    # Face detection
                    face_roi = frame[y1:y2, x1:x2]
                    faces = face_cascade.detectMultiScale(face_roi, 1.1, 5, minSize=(30, 30))

                    if len(faces) > 0:
                        see_tv_count += 1
                        current_see_tv_ids.add(current_id)
                        
                        if current_id not in see_tv_start_time:
                            see_tv_start_time[current_id] = time.time()

                        for (fx, fy, fw, fh) in faces:
                            cv2.rectangle(frame, (x1 + fx, y1 + fy), 
                                         (x1 + fx + fw, y1 + fy + fh), 
                                         (255, 0, 0), 2)
                            cv2.putText(frame, "Looking", (x1 + fx, y1 + fy - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Not Looking", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                    # Draw person box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Calculate viewing durations
    see_tv_duration = {}
    for person_id in list(see_tv_start_time.keys()):
        if person_id in current_see_tv_ids:
            see_tv_duration[person_id] = time.time() - see_tv_start_time[person_id]
        else:
            see_tv_duration[person_id] = time.time() - see_tv_start_time[person_id]
            del see_tv_start_time[person_id]

    # FPS calculation
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()

    # Display info
    cv2.putText(frame, f"Persons: {person_count}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Watching: {see_tv_count}", (10, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 110), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display durations
    y_offset = 150
    for person_id, duration in see_tv_duration.items():
        cv2.putText(frame, f"ID {person_id}: {duration:.1f}s", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y_offset += 30

    # Save to CSV
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if see_tv_count > 0:
            for person_id, duration in see_tv_duration.items():
                writer.writerow([person_id, person_count, see_tv_count, 
                               f"{duration:.2f}", id_device, timestamp])
        else:
            writer.writerow([current_id, person_count, see_tv_count, 
                           "0.00", id_device, timestamp])

    current_id += 1

    # Display
    cv2.imshow('YOLOv5 Monitoring', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()