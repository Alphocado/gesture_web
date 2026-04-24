# ✋ Gesture Recognition with Video Trigger

Proyek ini memungkinkan Anda untuk **mengenali gerakan tangan secara real‑time** dan memicu
pemutaran video berdasarkan gerakan yang terdeteksi.  
Dibangun dengan **MediaPipe Hands**, **TensorFlow/Keras**, dan **TensorFlow.js**.

---

## 📁 Struktur Proyek

```
.
├── data/                   # Folder untuk file CSV hasil rekaman
├── models/                 # Model terlatih dan metadata
│   ├── gesture_model.h5
│   ├── gesture_model.keras
│   ├── classes.txt
│   ├── norm_params.json
│   └── tfjs_model/         # Model untuk web (TensorFlow.js)
│       ├── model.json
│       └── group1-shard1of1.bin
├── videos/                 # Video untuk setiap gesture (misal: halo.mp4, oke.mp4)
├── record_data.py          # Script untuk merekam landmark tangan
├── train_model.py          # Script pelatihan model klasifikasi
├── convert_fixed.py        # Script konversi model ke TFJS (manual, andal)
├── convert_to_tfjs.py      # (opsional) konversi otomatis, kadang gagal di Windows
├── index.html              # Antarmuka web
├── script.js               # Logika utama web (kamera, prediksi, video)
├── style.css               # Tampilan antarmuka
├── requirements.txt        # Daftar dependensi Python
└── README.md               # Dokumentasi ini
```

---

## 🚀 Instalasi dan Persiapan Lingkungan

### 1. Clone / Download Proyek

Pastikan semua file proyek sudah berada dalam satu folder.

### 2. Buat Virtual Environment (venv)

**Windows (Command Prompt / PowerShell)**

```bash
python -m venv gesture_env
gesture_env\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv gesture_env
source gesture_env/bin/activate
```

### 3. Install Dependensi

Pastikan `requirements.txt` sudah berada di root proyek.  
Isinya ( **tanpa `tensorflowjs`** karena konversi akan dilakukan manual lewat `convert_fixed.py`):

```
mediapipe==0.10.9
opencv-python==4.8.1.78
tensorflow==2.15.0
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
h5py==3.9.0
protobuf==3.20.3
```

Jalankan instalasi:

```bash
pip install -r requirements.txt
```

> **⚠️ Penting:** Jangan upgrade NumPy ke versi 2.x – TensorFlow 2.15.0 hanya kompatibel dengan NumPy 1.x.

---

## 📹 1. Merekam Data Gesture

Gunakan `record_data.py` untuk mengumpulkan sampel gerakan tangan.  
Setiap sampel adalah file CSV yang berisi urutan landmark (126 nilai per frame) selama 3 detik.

**Perintah Dasar:**

```bash
python record_data.py --label halo --samples 30
```

- `--label` : Nama gesture (contoh: `halo`, `oke`).
- `--samples` : Jumlah sampel yang ingin direkam (default 30).
- `--duration`: Durasi tiap rekaman dalam detik (default 3.0).

**Cara Merekam:**

1. Jalankan perintah di atas.
2. Jendela kamera akan terbuka.
3. Tekan **SPASI** untuk memulai perekaman 3 detik.
4. Lakukan gerakan secara natural selama perekaman berlangsung.
5. Rekaman otomatis tersimpan di folder `data/` dengan nama `halo_YYYYMMDD_HHMMSS.csv`.
6. Ulangi hingga jumlah sampel tercapai.
7. Tekan **Q** untuk keluar.

### ✏️ Merekam dengan File Executable (.exe) – Tanpa Python

Jika tidak ingin menginstal Python di komputer perekam, Anda bisa membungkus
`record_data.py` menjadi **satu file .exe** dengan PyInstaller.

**Cara membuatnya (di komputer yang sudah ada Python + environment):**

```bash
pip install pyinstaller
pyinstaller --onefile --collect-all mediapipe --collect-all opencv-python --collect-all tensorflow --name record_data record_data.py
```

Hasilnya ada di folder `dist/record_data.exe`.  
Jalankan dari Command Prompt:

```bat
record_data.exe --label kicau_mania --samples 30
```

> 📌 **Catatan:**
>
> - File .exe berukuran besar (~1.5 GB) karena TensorFlow & MediaPipe ikut terbundel.
> - Jika muncul error `FileNotFoundError: The path does not exist`, ulangi build dengan opsi `--collect-all mediapipe`.
> - Di komputer target, pastikan [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) sudah terinstal.

---

## 🧠 2. Melatih Model Klasifikasi

Setelah data terkumpul, latih model dengan `train_model.py`.

**Jalankan:**

```bash
python train_model.py
```

**Apa yang terjadi?**

- Membaca semua file CSV di folder `data/`.
- Mengekstrak fitur **mean** dan **standard deviation** dari setiap sequence → vektor 252 dimensi.
- Membagi data latih (80%) dan uji (20%).
- Melatih neural network sederhana (Dense layers).
- Menyimpan:
  - `models/gesture_model.h5` (format Keras H5)
  - `models/gesture_model.keras` (format Keras v3)
  - `models/classes.txt` (daftar label)
  - `models/norm_params.json` (mean & std untuk normalisasi)

> ✏️ **Perhatian:**  
> `train_model.py` akan mencoba konversi TFJS otomatis di akhir.  
> **Jika gagal** (karena `tensorflowjs_converter` tidak terinstal), **abaikan saja**.  
> Kita akan menggunakan `convert_fixed.py` di langkah berikutnya.

---

## 🌐 3. Konversi Model ke TensorFlow.js

Karena `tensorflowjs` rawan konflik di Windows dan sengaja dikeluarkan dari
`requirements`, kita pakai script **`convert_fixed.py`** yang menggunakan TensorFlow langsung.

**Jalankan:**

```bash
python convert_fixed.py
```

**Hasil:**

- Folder `models/tfjs_model/` akan berisi:
  - `model.json` (arsitektur model)
  - `group1-shard1of1.bin` (bobot model)

Pastikan ukuran file `.bin` **tidak 0 byte** (minimal beberapa ratus KB).

---

## 🖥️ 4. Menjalankan Aplikasi Web

### a. Siapkan Video

Buat folder `videos/` di root proyek. Simpan file video dengan nama **persis sesuai
label** di `classes.txt`, misalnya:

```
videos/
├── halo.mp4
└── ok.mp4          # perhatikan nama persis "ok", bukan "oke"
```

### b. Jalankan Server Lokal

**Dengan `npx serve` (direkomendasikan):**

```bash
npx serve .
```

Lalu buka `http://localhost:3000`.

**Atau dengan Python:**

```bash
python -m http.server 8000
# Buka http://localhost:8000
```

### c. Cara Penggunaan di Web

1. Izinkan akses kamera.
2. Tampilkan telapak tangan di depan kamera.
3. Lakukan gerakan selama **3 detik** (buffer akan terisi).
4. Prediksi muncul, dan jika confidence ≥ 70% video terkait diputar.

---

## 🛠️ Troubleshooting – Error yang Sering Terjadi

### ❌ Error: `_ARRAY_API not found` / `numpy.core.umath failed to import`

**Penyebab:** NumPy 2.x terinstal, tetapi TensorFlow 2.15.0 masih butuh NumPy 1.x.  
**Solusi:**

```bash
pip install numpy==1.24.3 --force-reinstall
```

Pastikan environment bersih dari NumPy 2. Jika perlu, hapus venv lama dan buat baru.

---

### ❌ Error: `ModuleNotFoundError: No module named 'cv2'` / `'mediapipe'`

**Penyebab:** Environment belum terinstall semua dependensi.  
**Solusi:** Pastikan Anda sudah menjalankan `pip install -r requirements.txt` di dalam venv yang aktif.

---

### ❌ Error: `uvloop does not support Windows` saat install `tensorflowjs`

**Penyebab:** `tensorflowjs` versi baru membawa `uvloop` yang tidak mendukung Windows.  
**Solusi (sudah diterapkan di proyek ini):**

- Hapus `tensorflowjs` dari `requirements.txt`.
- Gunakan `convert_fixed.py` untuk konversi model (tidak perlu `tensorflowjs`).

---

### ❌ Error: `FileNotFoundError: The path does not exist` saat menjalankan `.exe`

**Penyebab:** Saat build PyInstaller, file model MediaPipe tidak ikut tersalin.  
**Solusi:** Build ulang dengan tambahan `--collect-all mediapipe`:

```bash
pyinstaller --onefile --collect-all mediapipe --collect-all opencv-python --collect-all tensorflow --name record_data record_data.py
```

---

### ❌ Error: "Based on the provided shape, … tensor should have X values but has 0"

**Penyebab:** File `group1-shard1of1.bin` kosong atau tidak lengkap.  
**Solusi:**

- Jalankan ulang `python convert_fixed.py`.
- Periksa output, pastikan total bytes > 0.
- Jika tetap 0, pastikan model H5 valid dengan menjalankan:
  ```python
  import tensorflow as tf
  model = tf.keras.models.load_model('models/gesture_model.h5')
  model.summary()
  ```

---

### ❌ Buffer lambat atau video tidak terpicu di HP

- **Buffer lebih dari 3 detik:** Buka `script.js`, ubah `BUFFER_DURATION_MS` menjadi `2000` (2 detik).
- **Confidence rendah:** Ubah `CONFIDENCE_THRESHOLD` di `script.js` menjadi `0.5`.
- **Video tidak ditemukan:** Pastikan nama file video **sama persis** dengan isi `models/classes.txt` (case‑sensitive).

---

### ❌ Kamera terlihat “tertarik” atau jelek di HP

Kamera sudah diatur untuk mengikuti resolusi asli. Jika masih bermasalah, pastikan
CSS menggunakan `object-fit: cover` atau `contain` pada video dan canvas.

---

## 📦 `requirements.txt` Lengkap (Final)

```
mediapipe==0.10.9
opencv-python==4.8.1.78
tensorflow==2.15.0
scikit-learn==1.3.2
pandas==2.0.3
numpy==1.24.3
h5py==3.9.0
protobuf==3.20.3
```

> **Catatan:** `tensorflowjs` tidak disertakan karena kita menggunakan `convert_fixed.py`.

---

## 🧪 Pengujian Mandiri

1. **Rekam data minimal 2 kelas** (misal: `halo` dan `ok`) masing‑masing 30 sampel.
2. **Latih model** → `python train_model.py`.
3. **Konversi model** → `python convert_fixed.py` (pastikan `.bin` > 0).
4. **Letakkan video** di folder `videos/` dengan nama yang sesuai.
5. **Jalankan server** (`npx serve .`) dan buka browser.
6. **Lakukan gerakan** selama 3 detik dan lihat video terputar.

---

**Selamat mencoba!** 🎉  
Proyek ini kini tahan terhadap berbagai error umum, dan bisa dijalankan di
perangkat Windows, macOS, maupun Linux dengan langkah‑langkah yang sama.

```

```
