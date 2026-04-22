"""
convert_to_tfjs.py  (v2 — fixed)
=================================
Mengatasi dua masalah sekaligus:
  1. tensorflowjs CLI rusak karena konflik jax
  2. File .bin kosong dari converter manual v1

Cara pakai:
    python convert_to_tfjs.py
"""

import os
import sys
import json
import types
import numpy as np

# ==========================================
# KONFIGURASI
# ==========================================
H5_PATH  = "models/gesture_model.h5"
OUT_DIR  = "models/tfjs_model"
BIN_NAME = "group1-shard1of1.bin"

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================================
# IMPORT TENSORFLOW
# ==========================================
try:
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__} terdeteksi")
except ImportError:
    print("❌ TensorFlow tidak ditemukan.")
    sys.exit(1)

# ==========================================
# LOAD MODEL
# ==========================================
print(f"\n📂 Memuat model dari '{H5_PATH}' ...")
if not os.path.exists(H5_PATH):
    print(f"❌ File tidak ditemukan: {H5_PATH}")
    sys.exit(1)

model = tf.keras.models.load_model(H5_PATH, compile=False)
print("✅ Model berhasil dimuat.")

# ==========================================
# COBA PENDEKATAN 1: Patch jax + pakai tensorflowjs API
# ==========================================
def try_with_tfjs_api():
    print("\n🔧 Pendekatan 1: Memperbaiki import jax yang rusak...")
    try:
        # Buat modul palsu untuk menghindari ImportError dari jax
        fake = types.ModuleType("tensorflowjs.converters.jax_conversion")
        fake.convert_jax = lambda *a, **kw: None
        sys.modules["tensorflowjs.converters.jax_conversion"] = fake

        import tensorflowjs as tfjs
        print("   ✅ tensorflowjs berhasil di-import (setelah patch)")

        tfjs.converters.save_keras_model(model, OUT_DIR)
        print(f"   ✅ Konversi via tfjs API berhasil → {OUT_DIR}/")
        return True
    except Exception as e:
        print(f"   ⚠️  Pendekatan 1 gagal: {e}")
        return False


# ==========================================
# COBA PENDEKATAN 2: Konversi manual (robust)
# ==========================================
def try_manual_conversion():
    print("\n🔧 Pendekatan 2: Konversi manual...")

    # Kumpulkan semua weights dengan nama yang benar
    weight_specs = []
    buffer = bytearray()

    print("   Mengekstrak weights:")
    for var in model.variables:          # .variables sudah termasuk semua (trainable + non-trainable)
        arr = var.numpy().astype(np.float32)

        # Bersihkan nama: hapus prefix nama model dan suffix ":0"
        name = var.name                  # misal: "gesture_model/dense/kernel:0"
        parts = name.split("/")
        # Jika ada prefix nama model (bukan 'dense', 'batch_normalization', dll), buang
        if len(parts) > 1 and parts[0] == model.name:
            name = "/".join(parts[1:])
        name = name.rstrip(":0").rstrip(":")

        spec = {
            "name":  name,
            "shape": list(arr.shape),
            "dtype": "float32",
        }
        weight_specs.append(spec)

        raw = arr.flatten().tobytes()
        buffer += raw

        n_vals = arr.size
        print(f"      ✓ {name:55s} shape={list(arr.shape)}")

    total_bytes = len(buffer)
    if total_bytes == 0:
        print("   ❌ Buffer kosong! Tidak ada weight yang ditemukan.")
        return False

    # Tulis file .bin
    bin_path = os.path.join(OUT_DIR, BIN_NAME)
    with open(bin_path, "wb") as f:
        f.write(buffer)
    print(f"\n   💾 {BIN_NAME} → {total_bytes:,} bytes")

    # Verifikasi ukuran
    expected = sum(int(np.prod(s["shape"])) * 4 for s in weight_specs)
    actual   = os.path.getsize(bin_path)
    if expected != actual:
        print(f"   ❌ Ukuran .bin salah! expected={expected}, actual={actual}")
        return False
    print(f"   ✅ Ukuran .bin verified: {actual:,} bytes")

    # Buat model.json
    topology = {
        "class_name": model.__class__.__name__,
        "config":     model.get_config(),
        "keras_version": tf.keras.__version__,
        "backend":    "tensorflow",
    }
    model_json = {
        "format":        "layers-model",
        "generatedBy":   f"keras {tf.keras.__version__}",
        "convertedBy":   "convert_to_tfjs.py v2",
        "modelTopology": topology,
        "weightsManifest": [{
            "paths":   [BIN_NAME],
            "weights": weight_specs,
        }],
    }

    json_path = os.path.join(OUT_DIR, "model.json")
    with open(json_path, "w") as f:
        json.dump(model_json, f, indent=2)
    print(f"   💾 model.json → {os.path.getsize(json_path):,} bytes")
    return True


# ==========================================
# JALANKAN — coba pendekatan 1, fallback ke 2
# ==========================================
success = try_with_tfjs_api()
if not success:
    success = try_manual_conversion()

if not success:
    print("\n❌ Semua pendekatan gagal. Coba:\n   pip install tensorflowjs==3.15.0")
    sys.exit(1)

# ==========================================
# LAPORAN AKHIR
# ==========================================
print("\n" + "=" * 55)
print("🎉 Konversi SELESAI!\n")
print(f"  {OUT_DIR}/")
for fn in sorted(os.listdir(OUT_DIR)):
    size = os.path.getsize(os.path.join(OUT_DIR, fn))
    print(f"  ├── {fn}  ({size:,} bytes)")

# Validasi minimal
json_path = os.path.join(OUT_DIR, "model.json")
with open(json_path) as f:
    mj = json.load(f)
n_weights = len(mj["weightsManifest"][0]["weights"])
print(f"\n  Total weight tensors : {n_weights}")
print(f"  Total binary size    : {os.path.getsize(os.path.join(OUT_DIR, BIN_NAME)):,} bytes")
print("\n✅ Sekarang buka ulang browser dengan: npx serve .")