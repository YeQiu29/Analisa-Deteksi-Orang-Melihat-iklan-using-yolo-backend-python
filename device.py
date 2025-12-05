import cv2
import torch
import csv
from datetime import datetime
from pathlib import Path
import platform
import socket
import time

# Load model YOLOv5 yang sudah ditraining
model_path = 'yolov5s.pt'  # Ganti dengan path model Anda
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

# Load detektor wajah Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi webcam
cap = cv2.VideoCapture(1)  # Ganti dengan indeks kamera eksternal

# Folder untuk menyimpan CSV
csv_folder = Path('csv')  # Ganti dengan path folder CSV Anda
csv_folder.mkdir(exist_ok=True)  # Buat folder jika belum ada

# File CSV
csv_file = csv_folder / 'detections.csv'

# Buat file CSV jika belum ada
if not csv_file.exists():
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'person_count', 'see_tv_count', 'id_device', 'timestamp'])

# Variabel untuk menyimpan ID unik
detected_ids = set()
current_id = 0

# Dapatkan informasi perangkat
def get_device_id():
    # Gunakan nama host sebagai ID perangkat
    hostname = socket.gethostname()
    return hostname

# Dapatkan ID perangkat
id_device = get_device_id()

# Threshold confidence score
conf_threshold = 0.3

# Variabel untuk menghitung FPS
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera. Pastikan kamera terhubung.")
        break

    # Lakukan deteksi dengan YOLOv5
    results = model(frame)

    # Ambil hasil deteksi dengan confidence score di atas threshold
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] > conf_threshold]  # Filter berdasarkan threshold

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

            if len(faces) > 0:
                # Jika wajah terdeteksi, anggap orang tersebut sedang melihat TV
                see_tv_count += 1
                for (fx, fy, fw, fh) in faces:
                    # Gambar bounding box wajah
                    cv2.rectangle(frame, (xmin + fx, ymin + fy), (xmin + fx + fw, ymin + fy + fh), (255, 0, 0), 2)
                    cv2.putText(frame, "Looking at TV", (xmin + fx, ymin + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # Jika wajah tidak terdeteksi, anggap orang tersebut tidak melihat TV
                cv2.putText(frame, "Not Looking", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            # Gambar bounding box person
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Hitung FPS
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1:  # Hitung FPS setiap 1 detik
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()

    # Tampilkan jumlah orang dan FPS di pojok kiri atas
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"See TV Count: {see_tv_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan frame
    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Simpan hasil deteksi ke CSV
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_id, person_count, see_tv_count, id_device, timestamp])

    # Berikan ID unik untuk setiap frame
    current_id += 1

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam dan tutup semua window
cap.release()
cv2.destroyAllWindows()