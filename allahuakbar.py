import cv2
import torch
import time

# Load model (pastikan file yolov5s.pt di path yang benar)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', force_reload=True)

# Set hanya deteksi class 'person' (cocok untuk model dengan 1 class)
model.conf = 0.4  # confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.classes = [0]  # index class 'person'

# Inisialisasi kamera (ganti index jika perlu, misal /dev/video1)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Tidak bisa membuka kamera")
    exit()

print("Mulai deteksi. Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    # Inference
    results = model(frame)

    # Visualisasi hasil
    annotated_frame = results.render()[0]  # render() mengembalikan list image

    # Tampilkan frame
    cv2.imshow("YOLOv5 Person Detection", annotated_frame)

    # Keluar jika tekan 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
