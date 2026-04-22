import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import subprocess
import sys

# ==========================================
# KONFIGURASI
# ==========================================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_DIR = "data"
MODEL_DIR = "models"
TFJS_MODEL_DIR = os.path.join(MODEL_DIR, "tfjs_model")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TFJS_MODEL_DIR, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================
def load_data():
    """Membaca semua CSV. Setiap file = satu sequence gerakan."""
    X = []
    y = []
    csv_files = glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"📂 Ditemukan {len(csv_files)} file data.")

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if 'label' not in df.columns:
                print(f"⚠️ Kolom 'label' tidak ada di {file}. Dilewati.")
                continue

            label = df['label'].iloc[0]
            landmarks = df.iloc[:, 1:].values.astype(np.float32)

            if landmarks.shape[0] < 5:
                print(f"⚠️ {file} hanya punya {landmarks.shape[0]} frame. Dilewati.")
                continue

            if landmarks.shape[1] != 126:
                print(f"⚠️ {file} punya {landmarks.shape[1]} kolom (bukan 126). Dilewati.")
                continue

            # Fitur: mean + std tiap koordinat = 126 * 2 = 252 nilai
            mean_vals = np.mean(landmarks, axis=0)
            std_vals = np.std(landmarks, axis=0)
            features = np.concatenate([mean_vals, std_vals])  # shape: (252,)

            X.append(features)
            y.append(label)
        except Exception as e:
            print(f"⚠️ Gagal membaca {file}: {e}")

    if len(X) == 0:
        raise ValueError("❌ Tidak ada data valid. Rekam data dulu dengan record_data.py")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    print(f"✅ Total sampel valid: {len(y)}")
    print(f"📏 Dimensi fitur: {X.shape[1]} (harus 252)")
    return X, y


print("=" * 50)
print("🔄 Memuat data...")
X, y = load_data()

# ==========================================
# ENCODE LABEL
# ==========================================
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)
print(f"🏷️ Kelas ({num_classes}): {list(le.classes_)}")

if num_classes < 2:
    raise ValueError(f"❌ Hanya {num_classes} kelas. Minimal 2 kelas untuk klasifikasi.")

# ==========================================
# SPLIT DATA
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=SEED, stratify=y_encoded
)
print(f"📊 Train: {len(X_train)} | Test: {len(X_test)}")

# ==========================================
# NORMALISASI
# ==========================================
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train_norm = (X_train - mean) / (std + 1e-8)
X_test_norm = (X_test - mean) / (std + 1e-8)

norm_params = {'mean': mean.tolist(), 'std': std.tolist()}
norm_path = os.path.join(MODEL_DIR, "norm_params.json")
with open(norm_path, "w") as f:
    json.dump(norm_params, f)
print(f"📊 Parameter normalisasi disimpan → {norm_path}")

# ==========================================
# BUILD MODEL
# ==========================================
model = keras.Sequential([
    layers.Input(shape=(X_train_norm.shape[1],), name="dense_input"),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
], name="gesture_model")

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# ==========================================
# TRAIN
# ==========================================
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-6),
]

history = model.fit(
    X_train_norm, y_train,
    validation_data=(X_test_norm, y_test),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

loss, acc = model.evaluate(X_test_norm, y_test, verbose=0)
print(f"\n📈 Akurasi Test: {acc * 100:.2f}%")
print(f"📉 Loss Test:    {loss:.4f}")

if acc < 0.7:
    print("⚠️  Akurasi rendah (<70%). Pertimbangkan untuk:")
    print("   - Menambah jumlah sampel per kelas")
    print("   - Memastikan gerakan jelas dan konsisten saat merekam")

# ==========================================
# SIMPAN MODEL
# ==========================================

# 1. Simpan sebagai format Keras (.keras) - paling stabil
keras_path = os.path.join(MODEL_DIR, "gesture_model.keras")
model.save(keras_path)
print(f"💾 Model Keras disimpan → {keras_path}")

# 2. Simpan sebagai H5 (untuk kompatibilitas & konversi TFJS)
h5_path = os.path.join(MODEL_DIR, "gesture_model.h5")
model.save(h5_path)
print(f"💾 Model H5 disimpan → {h5_path}")

# 3. Simpan daftar kelas
classes_path = os.path.join(MODEL_DIR, "classes.txt")
with open(classes_path, "w") as f:
    f.write("\n".join(le.classes_))
print(f"📋 Daftar kelas disimpan → {classes_path}")

# ==========================================
# KONVERSI KE TENSORFLOW.JS
# ==========================================
print("\n🌐 Mengonversi model ke TensorFlow.js...")
print(f"   Input: {h5_path}")
print(f"   Output: {TFJS_MODEL_DIR}")

# PENTING: Gunakan --input_format=keras (bukan tf_saved_model)
# agar menghasilkan LayersModel yang kompatibel dengan tf.loadLayersModel()
cmd = [
    sys.executable, "-m", "tensorflowjs_converter",
    "--input_format=keras",
    "--output_format=tfjs_layers_model",
    "--quantize_float16",          # Kompres ukuran model ~50%
    h5_path,
    TFJS_MODEL_DIR
]

try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(f"✅ Model TFJS berhasil disimpan → {TFJS_MODEL_DIR}")
    print(f"   File: {os.listdir(TFJS_MODEL_DIR)}")
except subprocess.CalledProcessError as e:
    print(f"❌ Konversi TFJS gagal.")
    print(f"   STDOUT: {e.stdout}")
    print(f"   STDERR: {e.stderr}")
    print("\n💡 Coba install ulang: pip install tensorflowjs")
    print("   Lalu jalankan manual:")
    print(f"   tensorflowjs_converter --input_format=keras --output_format=tfjs_layers_model {h5_path} {TFJS_MODEL_DIR}")
    sys.exit(1)

# ==========================================
# VALIDASI OUTPUT
# ==========================================
model_json = os.path.join(TFJS_MODEL_DIR, "model.json")
if os.path.exists(model_json):
    print(f"\n✅ Validasi: {model_json} ada.")
else:
    print(f"\n❌ PERINGATAN: {model_json} tidak ditemukan! Cek error di atas.")
    sys.exit(1)

print("\n" + "=" * 50)
print("🎉 Training selesai!")
print(f"   Akurasi: {acc * 100:.2f}%")
print(f"   Kelas: {list(le.classes_)}")
print(f"   Model TFJS: {TFJS_MODEL_DIR}/")
print("\n📁 Struktur file yang dibutuhkan untuk web:")
print("   models/")
print("   ├── tfjs_model/")
print("   │   ├── model.json")
print("   │   └── group1-shard1of1.bin (atau beberapa shard)")
print("   ├── classes.txt")
print("   └── norm_params.json")
print("   videos/")
print("   └── <nama_kelas>.mp4  (video untuk setiap gesture)")
print("=" * 50)