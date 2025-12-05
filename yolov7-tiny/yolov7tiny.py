import cv2
import torch
import numpy as np

# Path ke file model .pt
model_path = 'yolo7-tiny.pt'

try:
    # Muat checkpoint
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    # Akses objek model dari dictionary checkpoint
    model = checkpoint['model'].float()
    
    # Atur model ke mode evaluasi
    model.eval()
    
    # Inisialisasi kamera laptop (0 biasanya default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Tidak dapat mengakses kamera.")
        exit()
    
    # Fungsi untuk menampilkan hasil deteksi
    def display_detection(frame, results):
        # Loop setiap deteksi pada frame saat ini
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, box)
            label = f'{model.names[int(cls)]} {conf:.2f}'
            # Gambar kotak pembatasan hijau pada objek terdeteksi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Tulis label di atas kotak pembatasan
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Tidak dapat membaca frame dari kamera.")
            break
        
        # Konversi BGR ke RGB karena model biasanya memakai RGB input 
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame sesuai dengan input model (misalnya 640x640)
        frame_resized = cv2.resize(frame_rgb, (640, 640))
        
        # Convert to tensor dan normalize [0..1]
        img_tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # batch dim
        
        with torch.no_grad():
            # Lakukan prediksi dengan model pada frame RGB asli ukuran penuh 
            pred = model(img_tensor)[0]  # output prediksi
        
        # Decode output prediksi menjadi bounding box + label + confidence score
        # Ini adalah bagian yang mungkin perlu disesuaikan dengan implementasi model Anda
        # Berikut adalah contoh sederhana menggunakan Non-Maximum Suppression (NMS)
        conf_threshold = 0.25
        iou_threshold = 0.45
        
        # Filter deteksi berdasarkan confidence threshold
        scores = pred[:, 4] * pred[:, 5:].max(1).values
        boxes = pred[:, :4]
        classes = pred[:, 5:].argmax(1)
        
        # Apply Non-Maximum Suppression (NMS)
        keep = torchvision.ops.nms(boxes, scores, iou_threshold)
        
        # Filter hasil NMS
        boxes = boxes[keep]
        scores = scores[keep]
        classes = classes[keep]
        
        # Buat objek deteksi yang sesuai dengan format yang diharapkan
        results = {
            'xyxy': [boxes],
            'conf': scores,
            'cls': classes
        }
        
        # Tampilkan hasil deteksi pada frame BGR asli agar OpenCV bisa menampilkannya benar warna 
        display_detection(frame, results)
        
        # Tampilkan jendela hasil deteksi 
        # Tekan tombol 'q' untuk keluar dari loop/menutup program.
        cv2.imshow('YOLOv7-Tiny Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error saat memuat model: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
