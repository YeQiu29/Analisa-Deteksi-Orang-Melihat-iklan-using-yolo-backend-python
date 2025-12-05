import cv2
import torch
import csv
from datetime import datetime
from pathlib import Path
import platform
import socket
import time
import os
import requests
import pandas as pd  # Untuk konversi CSV ke JSON

# ========== Konfigurasi ========== #
API_URL = "https://ai.geogiven.tech/api/upload_datawajah"  # Ganti dengan URL endpoint API kamu

# Cek ketersediaan GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan perangkat: {device}")

# Load model YOLOv5 yang sudah ditraining
model_path = 'yolov5s.pt'  # Ganti dengan path model Anda
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False).to(device)

# Load detektor wajah Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Folder untuk menyimpan CSV
csv_folder = Path('csv')
csv_folder.mkdir(exist_ok=True)

# File CSV
csv_file = csv_folder / 'detections.csv'

# Cek apakah file CSV sudah ada
if csv_file.exists():
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if len(rows) > 1:
            last_id = int(rows[-1][0])
        else:
            last_id = 0
else:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'person_count', 'see_tv_count', 'see_tv_duration', 'id_device', 'timestamp'])
    last_id = 0

# Variabel ID unik
current_id = last_id + 1
see_tv_start_time = {}

# Dapatkan ID perangkat
id_device = socket.gethostname()
conf_threshold = 0.45

fps_start_time = time.time()
fps_frame_count = 0
fps = 0

# Untuk interval pengiriman CSV
last_sent_time = time.time()
send_interval = 60  # 1 menit

# ====== Fungsi untuk kirim CSV sebagai JSON ====== #
def send_csv_as_json(csv_path):
    try:
        df = pd.read_csv(csv_path)
        json_data = df.to_dict(orient='records')  # Konversi ke list of dicts (JSON array)
        response = requests.post(API_URL, json=json_data)
        if response.status_code == 200:
            print(f"[INFO] Kirim data ke API berhasil: {response.status_code}")
            return True
        else:
            print(f"[WARNING] Gagal kirim data: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"[ERROR] Gagal mengirim data: {e}")
        return False

# ========== Loop utama deteksi ========== #
while True:
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    results = model(frame)
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] > conf_threshold]

    person_count, see_tv_count = 0, 0
    current_see_tv_ids = set()

    for _, detection in detections.iterrows():
        if detection['name'] == 'person':
            person_count += 1
            xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            face_roi = frame[ymin:ymax, xmin:xmax]
            faces = face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                see_tv_count += 1
                current_see_tv_ids.add(current_id)

                if current_id not in see_tv_start_time:
                    see_tv_start_time[current_id] = time.time()

                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(frame, (xmin + fx, ymin + fy), (xmin + fx + fw, ymin + fy + fh), (255, 0, 0), 2)
                    cv2.putText(frame, "Looking at TV", (xmin + fx, ymin + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Looking", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    see_tv_duration = {}
    for person_id in list(see_tv_start_time.keys()):
        if person_id in current_see_tv_ids:
            see_tv_duration[person_id] = time.time() - see_tv_start_time[person_id]
        else:
            see_tv_duration[person_id] = time.time() - see_tv_start_time[person_id]
            del see_tv_start_time[person_id]

    fps_frame_count += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()

    cv2.putText(frame, f"Jumlah Orang: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Melihat TV: {see_tv_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    y_offset = 150
    for person_id, duration in see_tv_duration.items():
        cv2.putText(frame, f"Person {person_id}: {duration:.2f}s", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y_offset += 30

    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Simpan hasil ke CSV
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if see_tv_count > 0:
            for person_id, duration in see_tv_duration.items():
                writer.writerow([person_id, person_count, see_tv_count, f"{duration:.2f}", id_device, timestamp])
        else:
            writer.writerow([current_id, person_count, see_tv_count, "0.00", id_device, timestamp])

    # Kirim data setiap 1 menit
    if time.time() - last_sent_time >= send_interval:
        status = send_csv_as_json(csv_file)
        print(f"[STATUS] Pengiriman data: {status}")
        last_sent_time = time.time()

    current_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()