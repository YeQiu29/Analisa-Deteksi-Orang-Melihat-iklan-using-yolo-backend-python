# 🚀 Proyek Deteksi Kepala Real-Time dengan YOLO

Selamat datang di Proyek Deteksi Kepala! Aplikasi ini menggunakan model YOLO (You Only Look Once) untuk mendeteksi kepala (atau orang) secara real-time dari feed webcam. Proyek ini juga dilengkapi dengan API endpoint untuk menerima dan menyimpan data deteksi.

![Placeholder Gambar Deteksi](https://via.placeholder.com/800x450.png?text=Contoh+Deteksi+Kepala)
*(Catatan: Gambar di atas adalah placeholder. Anda dapat menggantinya dengan screenshot aplikasi Anda.)*

---

## ✨ Fitur Utama

- **Deteksi Real-Time**: Menggunakan OpenCV dan PyTorch untuk menangkap video dari webcam dan melakukan deteksi objek secara langsung.
- **Model YOLO**: Diimplementasikan dengan model YOLOv5, YOLOv7, dan YOLOv8 yang kuat dan efisien untuk akurasi tinggi.
- **Penghitungan Objek**: Secara otomatis menghitung jumlah kepala/orang yang terdeteksi dalam frame.
- **Visualisasi**: Menampilkan bounding box (kotak pembatas) dan label kepercayaan (confidence score) pada objek yang terdeteksi.
- **API Server**: Dilengkapi dengan server FastAPI untuk menerima data hasil deteksi dalam format JSON dari sistem lain.
- **Pencatatan Data**: Menyimpan data deteksi yang diterima oleh API ke dalam file log berformat JSON dan CSV.

---

## 🛠️ Teknologi yang Digunakan

- **Python**: Bahasa pemrograman utama.
- **OpenCV**: Untuk pemrosesan gambar dan video secara real-time.
- **PyTorch**: Sebagai framework deep learning untuk menjalankan model YOLO.
- **YOLO (v5, v7, v8)**: Model deteksi objek state-of-the-art.
- **FastAPI**: Untuk membangun API server yang cepat dan efisien.
- **Pandas & NumPy**: Untuk manipulasi data.

---

## ⚙️ Instalasi dan Pengaturan

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di komputer Anda.

### 1. Clone Repositori
```bash
git clone <URL_REPOSITORI_ANDA>
cd <NAMA_FOLDER_PROYEK>
```

### 2. Buat dan Aktifkan Virtual Environment
Sangat disarankan untuk menggunakan virtual environment.
```bash
# Membuat environment
python -m venv env

# Mengaktifkan di Windows
.\env\Scripts\activate

# Mengaktifkan di macOS/Linux
source env/bin/activate
```

### 3. Instal Dependensi
File `requirements.txt` berisi semua pustaka Python yang dibutuhkan.
```bash
pip install -r requirements.txt
```

### 4. Unduh Model YOLO
File `.gitignore` diatur untuk mengabaikan file model (`.pt`, `.onnx`) karena ukurannya yang besar. Anda perlu mengunduhnya secara manual dan meletakkannya di direktori yang sesuai.

- **YOLOv5**: Unduh `yolov5s.pt` dan letakkan di direktori utama.
- **YOLOv7**: Unduh `yolov7-tiny.pt` dan letakkan di direktori `yolov7-tiny`.
- **YOLOv8**: Unduh `yolov8s.pt` dan letakkan di direktori `yolov8`.

Anda dapat menemukan model-model ini di repositori resmi Ultralytics atau YOLO.

---

## ▶️ Cara Menjalankan

Proyek ini memiliki dua komponen utama yang bisa dijalankan.

### Menjalankan Deteksi Real-Time
Untuk menjalankan deteksi langsung dari webcam Anda, jalankan skrip `head.py`.
```bash
python head.py
```
Sebuah jendela akan muncul menampilkan feed dari webcam Anda dengan kotak-kotak deteksi di sekitar orang yang teridentifikasi. Tekan tombol 'q' untuk keluar.

### Menjalankan API Server
Untuk menjalankan server yang siap menerima data deteksi, gunakan uvicorn.
```bash
uvicorn main:app --reload
```
Server akan berjalan di `http://127.0.0.1:8000`. Anda dapat mengirimkan data JSON ke endpoint `/upload` menggunakan metode POST.

---
Terima kasih telah menggunakan proyek ini! Jangan ragu untuk berkontribusi.
