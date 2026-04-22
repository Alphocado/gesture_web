"""
convert_fixed.py
----------------
Konversi model Keras (.h5) ke TensorFlow.js dengan verifikasi ukuran.
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

H5_PATH = "models/gesture_model.h5"
OUT_DIR = "models/tfjs_model"
BIN_NAME = "group1-shard1of1.bin"

os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load model
print(f"Memuat {H5_PATH}...")
model = tf.keras.models.load_model(H5_PATH, compile=False)
print("Model dimuat.")

# 2. Kumpulkan semua variabel (weights + bias + batch norm params)
weight_specs = []
buffer = bytearray()

print("Mengumpulkan weights:")
for w in model.weights:
    # w.name contoh: "gesture_model/dense/kernel:0"
    name = w.name
    # Hapus ":0" di akhir
    if name.endswith(":0"):
        name = name[:-2]
    # Hapus prefix nama model jika ada (opsional, tapi untuk TFJS sebaiknya tanpa prefix)
    # Misal "gesture_model/dense/kernel" -> "dense/kernel"
    parts = name.split("/")
    if parts[0] == model.name:
        name = "/".join(parts[1:])
    
    arr = w.numpy().astype(np.float32)
    spec = {
        "name": name,
        "shape": list(arr.shape),
        "dtype": "float32"
    }
    weight_specs.append(spec)
    
    # Tulis ke buffer
    raw = arr.tobytes()
    buffer.extend(raw)
    
    print(f"  {name:40} shape={str(arr.shape):20} size={arr.size}")

total_bytes = len(buffer)
print(f"\nTotal bytes: {total_bytes}")

if total_bytes == 0:
    print("ERROR: Buffer kosong!")
    sys.exit(1)

# 3. Tulis file .bin
bin_path = os.path.join(OUT_DIR, BIN_NAME)
with open(bin_path, "wb") as f:
    f.write(buffer)
print(f"File .bin disimpan: {bin_path} ({total_bytes} bytes)")

# Verifikasi ukuran file
actual_size = os.path.getsize(bin_path)
if actual_size != total_bytes:
    print(f"ERROR: Ukuran file tidak sesuai! Expected {total_bytes}, got {actual_size}")
    sys.exit(1)

# 4. Buat model.json
# Dapatkan konfigurasi model
model_config = model.get_config()

# Untuk TFJS LayersModel, topology harus berisi class_name dan config
topology = {
    "class_name": "Sequential",  # model.__class__.__name__,
    "config": model_config
}

model_json = {
    "format": "layers-model",
    "generatedBy": f"TensorFlow {tf.__version__}",
    "convertedBy": "convert_fixed.py",
    "modelTopology": topology,
    "weightsManifest": [{
        "paths": [BIN_NAME],
        "weights": weight_specs
    }]
}

json_path = os.path.join(OUT_DIR, "model.json")
with open(json_path, "w") as f:
    json.dump(model_json, f, indent=2)
print(f"model.json disimpan: {json_path}")

print("\n✅ Konversi selesai. Pastikan ukuran .bin > 0.")