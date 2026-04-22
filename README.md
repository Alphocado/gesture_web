# InclusiveVision Web

**Penerjemah Bahasa Isyarat berbasis Browser** — Training local Python, deploy ke GitHub Pages.

---

## 📁 Struktur Folder

```
project/
├── index.html              ← Web app utama
├── gesture_videos.json     ← Mapping gesture → file video
├── model/                  ← Output training (TF.js format)
│   ├── model.json
│   ├── labels_array.json
│   └── group1-shard1of1.bin
├── videos/                 ← File video gesture (.mp4)
│   ├── halo.mp4
│   └── terima_kasih.mp4
├── data/                   ← Data training (TIDAK perlu di-deploy)
│   ├── halo/
│   │   ├── sample_000.npy
│   │   └── ...
│   └── terima_kasih/
├── collect_data.py         ← TIDAK perlu di-deploy
├── train_web.py            ← TIDAK perlu di-deploy
└── requirements_web.txt    ← TIDAK perlu di-deploy
```

---

## 🚀 Alur Kerja

### 1. Install dependensi Python

```bash
pip install -r requirements_web.txt
```

### 2. Kumpulkan data gesture

```bash
# Rekam 30 sample gesture "halo" (masing-masing 3 detik)
python collect_data.py --label halo --samples 30

# Rekam gesture lainnya
python collect_data.py --label terima_kasih --samples 30
python collect_data.py --label apa_kabar    --samples 30
```

Saat merekam:

- Hitung mundur 3 detik → **tunjukkan gerakan** selama 3 detik
- Ulangi sebanyak `--samples` kali
- Usahakan pencahayaan bagus dan tangan terlihat jelas

### 3. Training model

```bash
python train_web.py --epochs 60 --batch 16
```

Output:

- `model/` → model TF.js siap pakai di browser
- `model/labels_array.json` → daftar label
- `gesture_videos.json` → template mapping (edit sesuai video Anda)

### 4. Siapkan video

Buat folder `videos/` dan masukkan file video:

```
videos/
├── halo.mp4
├── terima_kasih.mp4
└── apa_kabar.mp4
```

Edit `gesture_videos.json` jika nama file berbeda:

```json
{
  "halo": "videos/halo.mp4",
  "terima_kasih": "videos/terima_kasih.mp4"
}
```

### 5. Test lokal

Karena menggunakan fetch (MediaPipe CDN + load model), butuh HTTP server kecil:

```bash
python -m http.server 8080
```

Buka browser: **http://localhost:8080**

### 6. Deploy ke GitHub Pages

Upload semua file **kecuali** `data/`, `collect_data.py`, `train_web.py`, `requirements_web.txt`.

Yang perlu di-upload:

- `index.html`
- `model/`
- `videos/`
- `gesture_videos.json`

---

## ⚙️ Cara Kerja Web App

```
Tangan terdeteksi
       ↓
Rekam 3 detik landmark
       ↓
Aggregate (mean + std)
       ↓
TF.js inference
       ↓
Confidence > 55%?
       ↓
Putar video sekali
       ↓
Cooldown → kembali idle
```

---

## 🛠️ Troubleshooting

**"Gagal load model"**
→ Pastikan folder `model/` ada dan sudah diisi (`python train_web.py` dulu)
→ Buka via `python -m http.server`, bukan double-click index.html

**"Gesture tidak dikenali"**
→ Tambah sampel training (minimal 30-50 per gesture)
→ Pastikan pencahayaan saat training sama dengan saat testing
→ Turunkan threshold di `CONFIG.CONFIDENCE_MIN` di index.html

**Kamera tidak muncul**
→ Izinkan akses kamera di browser
→ Coba kamera lain: `--camera 1`

**Video tidak muncul**
→ Periksa path di `gesture_videos.json`
→ Format video gunakan `.mp4` (H.264)

---

## 📊 Tips Akurasi

- Minimum **30 sampel per gesture**, idealnya 50-100
- Variasikan posisi tangan sedikit-sedikit saat rekam
- Pencahayaan merata (hindari backlight)
- Jarak kamera konsisten (~50-80 cm)
- Untuk gesture dinamis (gerakan berpindah), std-feature akan lebih informatif daripada mean saja
