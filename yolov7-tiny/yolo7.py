import cv2
import torch
import csv
from datetime import datetime
from pathlib import Path
import platform
import socket
import time
import os

# Cek ketersediaan GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Menggunakan perangkat: {device}")

# Load model YOLOv7-tiny yang sudah ditraining
model_path = 'yolov7-tiny.pt'  # Ganti dengan path model custom Anda
model = torch.load(model_path, map_location=device)['model'].float().to(device)  # Load custom YOLOv7-tiny model
model.eval()  # Set model ke evaluation mode

# Load detektor wajah Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inisialisasi webcam
cap = cv2.VideoCapture(0)  # Ganti dengan indeks kamera eksternal
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Turunkan resolusi untuk meningkatkan FPS
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Folder untuk menyimpan CSV
csv_folder = Path('csv')  # Ganti dengan path folder CSV Anda
csv_folder.mkdir(exist_ok=True)  # Buat folder jika belum ada

# File CSV
csv_file = csv_folder / 'detections.csv'

# Cek apakah file CSV sudah ada
if csv_file.exists():
    # Baca ID terakhir dari file CSV
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        rows = list(reader)
        if len(rows) > 1:  # Jika ada data selain header
            last_id = int(rows[-1][0])  # Ambil ID terakhir
        else:
            last_id = 0
else:
    # Buat file CSV baru dengan header
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'person_count', 'see_tv_count', 'see_tv_duration', 'id_device', 'timestamp'])
    last_id = 0

# Variabel untuk menyimpan ID unik
current_id = last_id + 1

# Variabel untuk menyimpan durasi melihat TV
see_tv_start_time = {}  # {id: start_time}

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

# Fungsi untuk memproses hasil deteksi YOLOv7
def process_detections(detections, frame):
    global current_id, see_tv_start_time
    
    person_count = 0
    see_tv_count = 0
    current_see_tv_ids = set()
    
    # Deteksi YOLOv7 mengembalikan tensor dengan format [x1, y1, x2, y2, conf, cls]
    for det in detections:
        if len(det):
            for *xyxy, conf, cls in reversed(det):
                if conf > conf_threshold:
                    # Konversi class index ke nama class (sesuaikan dengan model Anda)
                    class_name = model.names[int(cls)] if hasattr(model, 'names') else 'person'
                    
                    if class_name == 'person':  # Sesuaikan dengan nama class Anda
                        person_count += 1
                        
                        # Ambil koordinat bounding box
                        xmin, ymin, xmax, ymax = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        
                        # Crop area wajah dari bounding box
                        face_roi = frame[ymin:ymax, xmin:xmax]
                        
                        # Deteksi wajah dalam area tersebut
                        faces = face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                        
                        if len(faces) > 0:
                            # Jika wajah terdeteksi, anggap orang tersebut sedang melihat TV
                            see_tv_count += 1
                            current_see_tv_ids.add(current_id)
                            
                            # Jika orang ini belum tercatat sebelumnya, catat waktu mulai
                            if current_id not in see_tv_start_time:
                                see_tv_start_time[current_id] = time.time()
                            
                            for (fx, fy, fw, fh) in faces:
                                # Gambar bounding box wajah
                                cv2.rectangle(frame, (xmin + fx, ymin + fy), (xmin + fx + fw, ymin + fy + fh), (255, 0, 0), 2)
                                cv2.putText(frame, "Looking at TV", (xmin + fx, ymin + fy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            # Jika wajah tidak terdeteksi, anggap orang tersebut tidak melihat TV
                            cv2.putText(frame, "Not Looking", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                        # Gambar bounding box person
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    return person_count, see_tv_count, current_see_tv_ids

while True:
    # Baca frame dari webcam
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera. Pastikan kamera terhubung.")
        break
    
    # Konversi frame ke format yang sesuai untuk YOLOv7
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    
    # Lakukan deteksi dengan YOLOv7-tiny
    with torch.no_grad():
        detections = model(img)
    
    # Proses hasil deteksi
    person_count, see_tv_count, current_see_tv_ids = process_detections(detections, frame)
    
    # Hitung durasi melihat TV untuk setiap orang
    see_tv_duration = {}
    for person_id in list(see_tv_start_time.keys()):
        if person_id in current_see_tv_ids:
            # Jika orang masih melihat TV, hitung durasi sampai sekarang
            see_tv_duration[person_id] = time.time() - see_tv_start_time[person_id]
        else:
            # Jika orang tidak lagi melihat TV, hapus dari pencatatan
            see_tv_duration[person_id] = time.time() - see_tv_start_time[person_id]
            del see_tv_start_time[person_id]
    
    # Hitung FPS
    fps_frame_count += 1
    if time.time() - fps_start_time >= 1:  # Hitung FPS setiap 1 detik
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_frame_count = 0
        fps_start_time = time.time()
    
    # Tampilkan jumlah orang, FPS, dan durasi melihat TV di pojok kiri atas
    cv2.putText(frame, f"Person Count: {person_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"See TV Count: {see_tv_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Tampilkan durasi melihat TV untuk setiap orang
    y_offset = 150
    for person_id, duration in see_tv_duration.items():
        cv2.putText(frame, f"Person {person_id}: {duration:.2f}s", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        y_offset += 30
    
    # Tampilkan frame
    cv2.imshow('YOLOv7-tiny Real-Time Detection', frame)
    
    # Simpan hasil deteksi ke CSV
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if see_tv_count > 0:
            for person_id, duration in see_tv_duration.items():
                writer.writerow([person_id, person_count, see_tv_count, f"{duration:.2f}", id_device, timestamp])
        else:
            # Jika tidak ada yang melihat TV, set durasi ke 0.00s
            writer.writerow([current_id, person_count, see_tv_count, "0.00", id_device, timestamp])
    
    # Berikan ID unik untuk setiap frame
    current_id += 1
    
    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam dan tutup semua window
cap.release()
cv2.destroyAllWindows()