import cv2
import torch
import numpy as np
from pathlib import Path
import random
from models.experimental import attempt_load  # Dari repo YOLOv7 asli

# Konfigurasi
MODEL_PATH = 'yolov7-tiny.pt'
SOURCE = '0'  # Webcam
CONF_THRES = 0.3
IOU_THRES = 0.45
IMG_SIZE = 640

def load_model(model_path, device):
    """Load model dengan penanganan error yang lebih baik"""
    try:
        # Coba load dengan cara YOLOv7 resmi
        model = attempt_load(model_path, map_location=device)
        print("Model berhasil dimuat menggunakan attempt_load")
        return model
    except:
        try:
            # Fallback: load langsung dengan torch
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
            model = torch.load(model_path, map_location=device)['model'].float()
            model.eval()
            print("Model berhasil dimuat dengan safe globals")
            return model
        except Exception as e:
            print(f"Gagal memuat model: {e}")
            exit()

def detect(model, img, device):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(img)[0]
    
    # Gunakan NMS dari YOLOv7 jika tersedia
    if hasattr(model, 'non_max_suppression'):
        pred = model.non_max_suppression(pred, CONF_THRES, IOU_THRES)
    else:
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES)
    return pred

def non_max_suppression(prediction, conf_thres, iou_thres):
    """Implementasi NMS sederhana jika tidak ada dari YOLOv7"""
    # ... (sama seperti sebelumnya)
    return output

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Menggunakan device: {device}")
    
    # Load model
    model = load_model(MODEL_PATH, device)
    
    # Dapatkan nama kelas dan warna
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
    cap = cv2.VideoCapture(int(SOURCE) if SOURCE.isnumeric() else SOURCE)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        pred = detect(model, frame_resized, device)
        
        # Visualisasi hasil
        for det in pred:
            if det is not None and len(det):
                for *xyxy, conf, cls in det:
                    label = f'{names[int(cls)]} {conf:.2f}'
                    x1, y1, x2, y2 = map(int, xyxy)
                    color = colors[int(cls)]
                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame_resized, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow('YOLOv7-tiny Detection', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()