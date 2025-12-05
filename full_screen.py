import cv2
import torch
import csv
import json
import time
import socket
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ========== Konfigurasi ========== #
API_URL = "https://ai.geogiven.tech/api/upload_datawajah"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Menggunakan perangkat: {device}")

# Load YOLOv5 model
model_path = 'yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)

# Load Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Folder setup
csv_folder = Path('csv')
csv_folder.mkdir(exist_ok=True)
csv_file = csv_folder / 'detections.csv'

log_folder = Path('logging')
log_folder.mkdir(exist_ok=True)

# CSV initialization
if csv_file.exists():
    with open(csv_file, mode='r') as file:
        reader = list(csv.reader(file))
        last_id = int(reader[-1][0]) if len(reader) > 1 else 0
else:
    with open(csv_file, mode='w', newline='') as file:
        csv.writer(file).writerow(['id', 'person_count', 'see_tv_count', 'see_tv_duration', 'id_device', 'timestamp'])
    last_id = 0

# Variabel umum
current_id = last_id + 1
see_tv_start_time = {}
id_device = socket.gethostname()
conf_threshold = 0.45
fps_start_time = time.time()
fps_frame_count = 0
fps = 0
last_sent_time = time.time()
send_interval = 60  # seconds

# Hapus log lama (>1 hari)
def clean_old_logs():
    now = datetime.now()
    cutoff = now - timedelta(days=1)
    for log_file in log_folder.glob('*.json'):
        try:
            file_time = datetime.strptime(log_file.stem.replace('json_', ''), '%Y-%m-%d_%H-%M-%S')
            if file_time < cutoff:
                log_file.unlink()
                print(f"[INFO] Menghapus log lama: {log_file}")
        except ValueError:
            continue

# Fungsi kirim CSV sebagai JSON
def send_csv_as_json(csv_path):
    try:
        df = pd.read_csv(csv_path)
        json_data = df.to_dict(orient='records')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = log_folder / f'json_{timestamp}.json'

        with open(log_file, 'w') as f:
            payload = {
                    'deteksi_json': json_data
            }

        # Simpan log JSON ke file
        with open(log_file, 'w') as f:
            json.dump(payload, f, indent=4)

        # Kirim ke endpoint API
        response = requests.post(API_URL, json=payload)

        with open(log_file, 'r+') as f:
            log_data = json.load(f)
            log_data['response_status'] = response.status_code
            log_data['response_text'] = response.text
            f.seek(0)
            json.dump(log_data, f, indent=4)
            f.truncate()

        if response.ok:
            print(f"[INFO] Data berhasil dikirim: {response.status_code}")
            return True
        else:
            print(f"[WARNING] Gagal kirim: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = log_folder / f'json_{timestamp}.json'
        with open(log_file, 'w') as f:
            json.dump({
                'deteksi_json': [],
                'error': str(e),
                'device_id': id_device
            }, f, indent=4)
        return False

# ========== Main Loop ========== #
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Tidak dapat membaca kamera.")
        break

    results = model(frame)
    detections = results.pandas().xyxy[0]
    detections = detections[detections['confidence'] > conf_threshold]

    person_count, see_tv_count = 0, 0
    current_see_tv_ids = set()

    for _, detection in detections.iterrows():
        if detection['name'] == 'person':
            person_count += 1
            xmin, ymin, xmax, ymax = map(int, [detection['xmin'], detection['ymin'], detection['xmax'], detection['ymax']])
            face_roi = frame[ymin:ymax, xmin:xmax]
            faces = face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces):
                see_tv_count += 1
                current_see_tv_ids.add(current_id)
                see_tv_start_time.setdefault(current_id, time.time())

                for (fx, fy, fw, fh) in faces:
                    cv2.rectangle(frame, (xmin + fx, ymin + fy), (xmin + fx + fw, ymin + fy + fh), (255, 0, 0), 2)
                    cv2.putText(frame, "Looking at TV", (xmin + fx, ymin + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Not Looking", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    # Hitung durasi
    see_tv_duration = {}
    for pid in list(see_tv_start_time.keys()):
        see_tv_duration[pid] = time.time() - see_tv_start_time[pid]
        if pid not in current_see_tv_ids:
            del see_tv_start_time[pid]

    # FPS counter
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1:
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()

    # Tampilkan informasi
    cv2.putText(frame, f"Jumlah Orang: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Melihat TV: {see_tv_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Tampilkan durasi per orang
    y_offset = 150
    for pid, dur in see_tv_duration.items():
        cv2.putText(frame, f"Person {pid}: {dur:.2f}s", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_offset += 25

    # Perbesar ukuran window
    cv2.namedWindow('YOLOv5 Real-Time Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('YOLOv5 Real-Time Detection', 1280, 720)  # Ukuran window

    cv2.imshow('YOLOv5 Real-Time Detection', frame)

    # Simpan ke CSV
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file, 'a', newline='') as file:
        writer = csv.writer(file)
        if see_tv_count > 0:
            for pid, dur in see_tv_duration.items():
                writer.writerow([pid, person_count, see_tv_count, f"{dur:.2f}", id_device, timestamp])
        else:
            writer.writerow([current_id, person_count, see_tv_count, "0.00", id_device, timestamp])

    # Kirim data setiap interval
    if time.time() - last_sent_time >= send_interval:
        clean_old_logs()
        send_csv_as_json(csv_file)
        last_sent_time = time.time()

    current_id += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
