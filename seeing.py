import cv2
import torch
import numpy as np
from pathlib import Path

# Load model YOLOv5 untuk deteksi orang
model_path = 'yolov5s.pt'  # Ganti dengan path model Anda
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Load detektor wajah Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi webcam
cap = cv2.VideoCapture(1)  # Ganti dengan indeks kamera eksternal

# Variabel counting
person_count = 0
see_tv_count = 0

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera. Pastikan kamera terhubung.")
        break

    # Lakukan deteksi dengan YOLOv5
    results = model(frame)

    # Ambil hasil deteksi
    detections = results.pandas().xyxy[0]  # Format deteksi: xmin, ymin, xmax, ymax, confidence, class, name

    # Reset counting
    person_count = 0
    see_tv_count = 0

    for _, detection in detections.iterrows():
        if detection['name'] == 'person':  # Sesuaikan dengan nama class Anda
            person_count += 1

            # Ambil koordinat bounding box
            xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])

            # Crop area wajah dari bounding box
            face_roi = frame[ymin:ymax, xmin:xmax]

            # Deteksi wajah dalam area tersebut
            faces = face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (fx, fy, fw, fh) in faces:
                # Gambar bounding box wajah
                cv2.rectangle(frame, (xmin + fx, ymin + fy), (xmin + fx + fw, ymin + fy + fh), (255, 0, 0), 2)

                # Estimasi orientasi wajah (sederhana: asumsikan wajah menghadap kamera jika lebar wajah > threshold)
                if fw > 50:  # Threshold lebar wajah
                    see_tv_count += 1
                    cv2.putText(frame, "Looking at TV", (xmin + fx, ymin + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Not Looking", (xmin + fx, ymin + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Gambar bounding box person
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Tampilkan counting di pojok kiri atas
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"See TV Count: {see_tv_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam dan tutup semua window
cap.release()
cv2.destroyAllWindows()