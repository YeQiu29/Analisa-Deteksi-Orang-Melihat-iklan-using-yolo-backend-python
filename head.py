import cv2
import torch
from pathlib import Path

# Load model YOLOv5 yang sudah ditraining
model_path = 'yolov5s.pt'  # Ganti dengan path model Anda
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Inisialisasi webcam
cap = cv2.VideoCapture(1)  # 0 untuk webcam default

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Lakukan deteksi dengan YOLOv5
    results = model(frame)

    # Ambil hasil deteksi
    detections = results.pandas().xyxy[0]  # Format deteksi: xmin, ymin, xmax, ymax, confidence, class, name

    # Hitung jumlah orang yang terdeteksi
    person_count = 0
    for _, detection in detections.iterrows():
        if detection['name'] == 'person':  # Sesuaikan dengan nama class Anda
            person_count += 1

            # Gambar bounding box dan label
            xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"person {detection['confidence']:.2f}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Tampilkan jumlah orang di pojok kiri atas
    cv2.putText(frame, f"person count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam dan tutup semua window
cap.release()
cv2.destroyAllWindows()