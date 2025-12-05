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

# Folder untuk log status API (ukuran kecil)
logging_data_folder = Path('logging_data')
logging_data_folder.mkdir(exist_ok=True)

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
send_interval = 60  # Kirim data setiap 60 detik

# ========== Fungsi Utama ========== #
def clean_old_logs(log_directory, days_threshold=3):
    """Hapus log file yang lebih tua dari `days_threshold`"""
    now = datetime.now()
    cutoff = now - timedelta(days=days_threshold)
    
    for log_file in log_directory.glob('*.log'):
        try:
            file_time_str = log_file.stem.replace('log_', '')
            file_time = datetime.strptime(file_time_str, '%Y-%m-%d_%H-%M-%S')
            if file_time < cutoff:
                log_file.unlink()
                print(f"[INFO] Menghapus log lama: {log_file}")
        except ValueError:
            continue

def log_api_status(status, message, response_code=None):
    """Catat status pengiriman API (ukuran kecil)"""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logging_data_folder / f'log_{timestamp}.log'
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': status,
        'message': message,
        'response_code': response_code
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    # Bersihkan log lama
    clean_old_logs(logging_data_folder)

def send_csv_as_json(csv_path):
    try:
        df = pd.read_csv(csv_path)
        
        # Pastikan tidak ada nilai NaN/Null
        df = df.where(pd.notnull(df), None)
        
        # Konversi ke format yang diharapkan API
        payload = {
            "deteksi_json": df.to_dict(orient='records')
        }

        # Debug: Tampilkan payload sebelum dikirim
        print("[DEBUG] Payload yang akan dikirim:", json.dumps(payload, indent=2))

        response = requests.post(
            API_URL,
            json=payload,  # Gunakan json= bukan data=
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 200:
            print("[INFO] Data berhasil dikirim!")
            return True
        else:
            print(f"[ERROR] Server merespons {response.status_code}: {response.text}")
            return False
            
    except Exception as e:
        print(f"[CRITICAL] Error: {str(e)}")
        return False

# ========== Main Loop ========== #
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Tidak dapat membaca frame dari kamera.")
            break

        # Deteksi objek dengan YOLOv5
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

        # Hitung durasi melihat TV
        see_tv_duration = {}
        for pid in list(see_tv_start_time.keys()):
            see_tv_duration[pid] = time.time() - see_tv_start_time[pid]
            if pid not in current_see_tv_ids:
                del see_tv_start_time[pid]

        # Hitung FPS
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()

        # Tampilkan informasi di layar
        cv2.putText(frame, f"Orang: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Melihat TV: {see_tv_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Simpan data ke CSV
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            if see_tv_count > 0:
                for pid, dur in see_tv_duration.items():
                    writer.writerow([pid, person_count, see_tv_count, f"{dur:.2f}", id_device, timestamp])
            else:
                writer.writerow([current_id, person_count, see_tv_count, "0.00", id_device, timestamp])

        # Kirim data ke API setiap `send_interval` detik
        if time.time() - last_sent_time >= send_interval:
            send_csv_as_json(csv_file)
            last_sent_time = time.time()

        current_id += 1

        # Tampilkan frame
        cv2.imshow('YOLOv5 Real-Time Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_api_status(True, "Program dihentikan oleh pengguna")
            break

except Exception as e:
    error_msg = f"Terjadi kesalahan: {str(e)}"
    print(f"[CRITICAL] {error_msg}")
    log_api_status(False, error_msg)
finally:
    cap.release()
    cv2.destroyAllWindows()