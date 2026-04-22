#!/usr/bin/env python3
"""
train_web.py – Train model dari data yang dikumpulkan, lalu export ke TF.js.

Cara pakai:
  python train_web.py --epochs 50 --batch 16

Prerequisite:
  pip install mediapipe opencv-python tensorflow scikit-learn tensorflowjs

Output:
  model/           → folder model TF.js (untuk web)
  labels.json      → mapping label → index
  gesture_videos.json → mapping label → path video (perlu diisi manual)
"""

import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter

# ── CONFIG ────────────────────────────────────────────────────────────────────
FEAT_DIM       = 126     # 2 tangan × 21 × 3
TARGET_FRAMES  = 45      # jumlah frame per sample (3 detik × 15 fps)
DATA_DIR       = "data"
MODEL_DIR      = "model"
LABELS_FILE    = "labels.json"
VIDEOS_MAP     = "gesture_videos.json"


# ── FEATURE AGGREGATION ───────────────────────────────────────────────────────
def aggregate_sequence(arr):
    """
    Ubah sequence frames (N, 126) → satu feature vector untuk MLP.
    Menggunakan mean + std → 252 features total.
    Mean menangkap posisi rata-rata, std menangkap gerakan/variasi.
    """
    mean = np.mean(arr, axis=0)          # (126,)
    std  = np.std(arr, axis=0)           # (126,)
    return np.concatenate([mean, std])   # (252,)


# ── MODEL ─────────────────────────────────────────────────────────────────────
def build_model(input_dim, n_classes):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dense(n_classes, activation='softmax'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(f"❌ Folder '{DATA_DIR}' tidak ditemukan. Jalankan collect_data.py dulu.")

    label_dirs = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    ])
    if not label_dirs:
        raise SystemExit(f"❌ Tidak ada subfolder di '{DATA_DIR}'.")

    print(f"📂 Label ditemukan: {label_dirs}")

    X_list, y_list = [], []
    for label in label_dirs:
        label_path = os.path.join(DATA_DIR, label)
        npy_files  = sorted([
            f for f in os.listdir(label_path) if f.endswith(".npy")
        ])
        if not npy_files:
            print(f"   ⚠️  '{label}' tidak punya file .npy – dilewati.")
            continue

        print(f"   '{label}' → {len(npy_files)} samples")
        for fname in npy_files:
            arr  = np.load(os.path.join(label_path, fname))   # (N_frames, 126)
            feat = aggregate_sequence(arr)                      # (252,)
            X_list.append(feat)
            y_list.append(label)

    if not X_list:
        raise SystemExit("❌ Tidak ada data valid.")

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list)
    return X, y, label_dirs


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # load data
    X, y, label_dirs = load_data()
    print(f"\n✔ Total samples: {len(X)} | Feature dim: {X.shape[1]}")
    print(f"  Distribusi: {dict(Counter(y))}")

    # encode label
    labels_map   = {lab: i for i, lab in enumerate(label_dirs)}
    labels_rev   = {i: lab for lab, i in labels_map.items()}
    y_idx        = np.array([labels_map[l] for l in y], dtype=np.int32)
    n_classes    = len(labels_map)
    y_cat        = tf.keras.utils.to_categorical(y_idx, num_classes=n_classes)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_idx)

    # class weights
    y_train_idx  = np.argmax(y_train, axis=1)
    cw_vals      = compute_class_weight('balanced',
                                        classes=np.unique(y_train_idx),
                                        y=y_train_idx)
    cw = {i: float(v) for i, v in enumerate(cw_vals)}
    print(f"\n  Class weights: {cw}")

    # train
    model = build_model(X.shape[1], n_classes)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-6),
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=args.epochs,
        batch_size=args.batch,
        class_weight=cw,
        callbacks=callbacks,
    )

    # evaluate
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n📊 Test loss: {loss:.4f} | Test acc: {acc:.4f}")

    # save labels.json
    with open(LABELS_FILE, "w") as f:
        json.dump(labels_map, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {LABELS_FILE}")

    # save labels dalam format array juga (lebih mudah di-load di JS)
    labels_array = [labels_rev[i] for i in range(n_classes)]
    labels_array_file = os.path.join(MODEL_DIR, "labels_array.json")
    with open(labels_array_file, "w") as f:
        json.dump(labels_array, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved {labels_array_file}")

    # buat gesture_videos.json template jika belum ada
    if not os.path.exists(VIDEOS_MAP):
        video_map = {label: f"videos/{label}.mp4" for label in label_dirs}
        with open(VIDEOS_MAP, "w") as f:
            json.dump(video_map, f, indent=2, ensure_ascii=False)
        print(f"✅ Template {VIDEOS_MAP} dibuat – isi path video sesuai file Anda!")
    else:
        print(f"ℹ️  {VIDEOS_MAP} sudah ada – tidak ditimpa.")

    # export ke TF.js
    print(f"\n🔄 Export ke TF.js di folder '{MODEL_DIR}/'...")
    try:
        import tensorflowjs as tfjs
        tfjs.converters.save_keras_model(model, MODEL_DIR)
        print(f"✅ TF.js model tersimpan di '{MODEL_DIR}/'")
    except ImportError:
        print("⚠️  tensorflowjs belum terinstall.")
        print("   Jalankan: pip install tensorflowjs")
        print("   Lalu jalankan konversi manual:")
        tmp_h5 = os.path.join(MODEL_DIR, "model_temp.h5")
        model.save(tmp_h5)
        print(f"   Model H5 disimpan di: {tmp_h5}")
        print(f"   Konversi: tensorflowjs_converter --input_format keras {tmp_h5} {MODEL_DIR}/")

    print("\n🎉 Selesai! Sekarang buka index.html di browser untuk testing.")
    print("   Pastikan folder 'videos/' berisi file video sesuai gesture_videos.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch",  type=int, default=16)
    main(ap.parse_args())