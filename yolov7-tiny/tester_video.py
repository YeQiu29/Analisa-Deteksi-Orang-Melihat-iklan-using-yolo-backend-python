import torch
import cv2
import sys
from pathlib import Path

# Setup path ke folder yolov7
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / 'yolov7'
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ==== Import dari repo yolov7 ====
from models.yolo import Model
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device
from utils.plots import plot_one_box
from utils.datasets import letterbox

# ==== Konfigurasi ====
weights_path = 'yolov7-tiny.pt'                  # path ke model hasil training
cfg_path = ROOT / 'cfg' / 'deploy' / 'yolov7-tiny.yaml'
imgsz = 640
class_names = ['person']                         # 1 kelas

# ==== Siapkan perangkat ====
device = select_device('')
half = device.type != 'cpu'

# ==== Load model dari config dan state_dict ====
model = Model(cfg_path, ch=3, nc=1)  # ch: channel, nc: number of classes
ckpt = torch.load(weights_path, map_location=device)
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
model.to(device).eval()
if half:
    model.half()

# ==== Buka kamera ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Gagal membuka kamera.")
    exit()

print("✅ Kamera dibuka. Tekan 'q' untuk keluar.")

# ==== Deteksi real-time ====
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame.")
        break

    # Preprocessing
    img = letterbox(frame, new_shape=imgsz)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR ke RGB, HWC ke CHW
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    # Visualisasi deteksi
    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{class_names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

    # Tampilkan hasil
    cv2.imshow("YOLOv7 Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
