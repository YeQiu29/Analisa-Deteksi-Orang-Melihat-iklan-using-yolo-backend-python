from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8s.pt")

# Threshold confidence
CONF_THRESHOLD = 0.5

# Buka kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()

print("✅ Kamera berhasil dibuka. Tekan 'q' untuk keluar.")

# Untuk penghitungan FPS
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Hitung waktu mulai
    start_time = time.time()

    # Deteksi objek
    results = model(frame, stream=True)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD:
                continue  # Lewati yang di bawah threshold

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hitung FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Tampilkan FPS di kiri atas
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    cv2.imshow("YOLOv8 Detection (Threshold + FPS)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
