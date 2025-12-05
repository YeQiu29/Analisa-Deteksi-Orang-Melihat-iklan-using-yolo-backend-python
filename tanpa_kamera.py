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

# Tambahkan folder logging_data untuk monitoring API
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
send_interval = 60  # seconds

# Fungsi untuk membersihkan log lama
def clean_old_logs(log_directory, days_threshold):
    """Hapus log file yang lebih tua dari days_threshold"""
    now = datetime.now()
    cutoff = now - timedelta(days=days_threshold)
    
    for log_file in log_directory.glob('*.log'):
        try:
            # Ekstrak timestamp dari nama file (format: log_YYYY-MM-DD_HH-MM-SS.log)
            file_time_str = log_file.stem.replace('log_', '')
            file_time = datetime.strptime(file_time_str, '%Y-%m-%d_%H-%M-%S')
            
            if file_time < cutoff:
                log_file.unlink()
                print(f"[INFO] Menghapus log lama: {log_file}")
        except ValueError as e:
            print(f"[WARNING] Gagal memproses file log {log_file}: {str(e)}")
            continue

# Fungsi untuk mencatat status pengiriman API
def log_api_status(status, message, response_code=None):
    """Catat status pengiriman API ke file log"""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file = logging_data_folder / f'log_{timestamp}.log'
    
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'status': status,
        'message': message,
        'device_id': id_device,
        'response_code': response_code
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_entry, f, indent=4)
    
    # Bersihkan log yang sudah lebih dari 3 hari
    clean_old_logs(logging_data_folder, 3)

# Fungsi kirim CSV sebagai JSON
def send_csv_as_json(csv_path):
    try:
        df = pd.read_csv(csv_path)
        json_data = df.to_dict(orient='records')
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_file = log_folder / f'json_{timestamp}.json'

        # Persiapkan payload
        payload = {
            'deteksi_json': json_data,
            'device_id': id_device,
            'timestamp': timestamp
        }

        # Simpan log JSON ke file
        with open(log_file, 'w') as f:
            json.dump(payload, f, indent=4)

        # Kirim ke endpoint API
        response = requests.post(API_URL, json=payload)
        response_data = {
            'status_code': response.status_code,
            'response_text': response.text
        }

        # Update log file dengan response
        with open(log_file, 'r+') as f:
            log_data = json.load(f)
            log_data.update(response_data)
            f.seek(0)
            json.dump(log_data, f, indent=4)
            f.truncate()

        if response.ok:
            msg = f"Data berhasil dikirim: {response.status_code}"
            print(f"[INFO] {msg}")
            log_api_status(True, msg, response.status_code)
            return True
        else:
            msg = f"Gagal kirim: {response.status_code} - {response.text}"
            print(f"[WARNING] {msg}")
            log_api_status(False, msg, response.status_code)
            return False

    except Exception as e:
        error_msg = f"Error saat mengirim data: {str(e)}"
        print(f"[ERROR] {error_msg}")
        
        # Buat file log error
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        error_log_file = log_folder / f'error_{timestamp}.json'
        with open(error_log_file, 'w') as f:
            json.dump({
                'error': str(e),
                'device_id': id_device,
                'timestamp': timestamp
            }, f, indent=4)
        
        log_api_status(False, error_msg)
        return False

# ========== Main Loop ========== #
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Tidak dapat membaca kamera.")
            log_api_status(False, "Tidak dapat membaca frame dari kamera")
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

        # Hitung durasi
        see_tv_duration = {}
        for pid in list(see_tv_start_time.keys()):
            see_tv_duration[pid] = time.time() - see_tv_start_time[pid]
            if pid not in current_see_tv_ids:
                del see_tv_start_time[pid]

        # FPS counter (opsional, bisa dihapus jika tidak diperlukan)
        fps_frame_count += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_frame_count / (time.time() - fps_start_time)
            fps_frame_count = 0
            fps_start_time = time.time()
            print(f"[INFO] FPS: {fps:.2f} | Orang: {person_count} | Melihat TV: {see_tv_count}")

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
            clean_old_logs(log_folder, 1)  # Bersihkan log JSON yang lebih dari 1 hari
            send_csv_as_json(csv_file)
            last_sent_time = time.time()

        current_id += 1

        # Hentikan program dengan keyboard interrupt (Ctrl+C)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_api_status(True, "Program dihentikan oleh pengguna")
            break

except KeyboardInterrupt:
    log_api_status(True, "Program dihentikan oleh keyboard interrupt")
    print("[INFO] Program dihentikan oleh user (Ctrl+C)")
except Exception as e:
    error_msg = f"Terjadi kesalahan: {str(e)}"
    print(f"[CRITICAL] {error_msg}")
    log_api_status(False, error_msg)
finally:
    cap.release()
    cv2.destroyAllWindows()