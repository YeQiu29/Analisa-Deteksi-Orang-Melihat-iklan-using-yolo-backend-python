import cv2
import os
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
#API_URL = "https://localhost:8000"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Menggunakan perangkat: {device}")

# Load model dan inisialisasi
model_path = 'yolov5s.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path).to(device)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Setup kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Dapatkan direktori tempat script berada
script_dir = os.path.dirname(os.path.abspath(__file__))

# Setup file dan folder - sekarang menggunakan path absolut
csv_folder = Path(script_dir) / 'csv'
csv_folder.mkdir(exist_ok=True)
csv_file = csv_folder / 'detections.csv'
logging_data_folder = Path(script_dir) / 'logging_data'
logging_data_folder.mkdir(exist_ok=True)

# Inisialisasi CSV
if csv_file.exists():
    with open(csv_file, mode='r') as file:
        reader = list(csv.reader(file))
        last_id = int(reader[-1][0]) if len(reader) > 1 else 0
else:
    # Inisialisasi CSV - Selalu buat file baru saat program mulai
    with open(csv_file, mode='w', newline='') as file:
        csv.writer(file).writerow(['id', 'person_count', 'see_tv_count', 'see_tv_duration', 'id_device', 'timestamp'])
    last_id = 0
    current_id = 1

# Variabel
current_id = last_id + 1
see_tv_start_time = {}
id_device = socket.gethostname()
conf_threshold = 0.45
fps_start_time = time.time()
fps_frame_count = 0
fps = 0
last_sent_time = time.time()
send_interval = 60  # 60 detik = 1 menit
backup_clean_interval = 3600  # 3600 detik = 1 jam
last_clean_time = time.time()

def clean_old_files(directory, pattern, hours_threshold=1):
    """Hapus file yang lebih tua dari jam_threshold"""
    now = time.time()
    cutoff = now - (hours_threshold * 3600)
    
    # Pastikan directory adalah Path object absolut
    directory = Path(directory)
    
    for file_path in directory.glob(pattern):
        if file_path.stat().st_mtime < cutoff:
            try:
                file_path.unlink()
                print(f"[CLEANER] Menghapus file lama: {file_path.name}")
            except Exception as e:
                print(f"[CLEANER ERROR] Gagal menghapus {file_path.name}: {str(e)}")

def log_api_status(status, message, response_code=None):
    """Catat status pengiriman API"""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logging_data_folder / f'log_{timestamp}.log'
    
    log_entry = {
        'status': status,
        'message': message,
        'response_code': response_code,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=2)
    
    # Bersihkan log API yang lama (3 hari)
    clean_old_files(logging_data_folder, "*.log", hours_threshold=72)

def send_csv_as_json(csv_path):
    """Kirim data dan reset CSV setelah berhasil"""
    global current_id
    
    try:
        df = pd.read_csv(csv_path)
        if len(df) == 0:
            print("[INFO] Tidak ada data baru untuk dikirim")
            return True

        payload = {
            "deteksi_json": df.to_dict(orient='records')
        }

        # Debug: Cek ukuran payload
        payload_size = len(json.dumps(payload)) / (1024 * 1024)  # Dalam MB
        print(f"[DEBUG] Ukuran payload: {payload_size:.2f} MB")

        response = requests.post(
            API_URL,
            json=payload,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )

        if response.status_code == 200:
            # Backup data yang sudah dikirim
            backup_filename = f"detections_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            backup_path = csv_folder / backup_filename
            df.to_csv(backup_path, index=False)
            
            # Reset CSV utama
            with open(csv_path, 'w', newline='') as file:
                csv.writer(file).writerow(['id', 'person_count', 'see_tv_count', 'see_tv_duration', 'id_device', 'timestamp'])
            
            # Reset variabel
            current_id = 1
            see_tv_start_time.clear()
            
            msg = f"Berhasil mengirim {len(df)} data | Reset CSV"
            print(f"[INFO] {msg}")
            log_api_status(True, msg, response.status_code)
            return True
        else:
            msg = f"Gagal kirim: {response.status_code} - {response.text}"
            print(f"[ERROR] {msg}")
            log_api_status(False, msg, response.status_code)
            return False

    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(f"[CRITICAL] {error_msg}")
        log_api_status(False, error_msg, 500)
        return False

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Tidak dapat membaca frame dari kamera.")
            break

        # Proses deteksi
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

        # Hitung durasi dan FPS
        see_tv_duration = {pid: time.time() - see_tv_start_time[pid] for pid in current_see_tv_ids}
        
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()

        # Display info
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

        # Kirim data setiap 1 menit
        if time.time() - last_sent_time >= send_interval:
            if send_csv_as_json(csv_file):
                print("[INFO] Data berhasil dikirim dan direset")
            last_sent_time = time.time()

        # Bersihkan backup file setiap 1 jam
        if time.time() - last_clean_time >= backup_clean_interval:
            clean_old_files(csv_folder, "detections_backup_*.csv", hours_threshold=1)
            last_clean_time = time.time()

        current_id += 1

        # Tampilkan frame
        cv2.imshow('Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    error_msg = f"Terjadi kesalahan: {str(e)}"
    print(f"[CRITICAL] {error_msg}")
    log_api_status(False, error_msg, 500)
finally:
    cap.release()
    cv2.destroyAllWindows()