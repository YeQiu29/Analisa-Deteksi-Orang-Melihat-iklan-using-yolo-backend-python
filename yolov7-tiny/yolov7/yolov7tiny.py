import cv2
import torch
import time

def main():
    # Load model
    model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7-tiny.pt')
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    
    # Kamera
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        results = model(frame)
        
        # Render results
        rendered_frame = results.render()[0]
        
        # Tampilkan
        cv2.imshow('YOLOv7-tiny Detection', rendered_frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()