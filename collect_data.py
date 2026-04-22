#!/usr/bin/env python3
"""
collect_data.py – Kumpulkan data gesture 3 detik untuk training web model.

Cara pakai:
  python collect_data.py --label halo --samples 30
  python collect_data.py --label terima_kasih --samples 30

Setiap sample = 3 detik rekaman → ekstrak landmark MediaPipe per frame.
Data disimpan di: data/<label>/sample_XXX.npy
"""

import os
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp

# ── CONFIG ────────────────────────────────────────────────────────────────────
RECORD_SECONDS  = 3      # durasi rekaman per sample
TARGET_FPS      = 15     # frame per detik yang di-sample
MAX_HANDS       = 2
PER_HAND        = 21 * 3
FEAT_DIM        = PER_HAND * MAX_HANDS   # 126


# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

def normalize_hand(landmarks):
    """
    Normalisasi 21 landmark relatif ke wrist (landmark[0]) dan scale.
    Return: array float32 shape (63,)
    """
    arr = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    origin = arr[0].copy()
    arr -= origin
    scale = np.max(np.linalg.norm(arr, axis=1))
    if scale > 0:
        arr /= scale
    return arr.flatten()   # (63,)


def frame_to_feature(results):
    """
    Ubah hasil MediaPipe satu frame → feature vector (126,).
    Jika tidak ada tangan → zeros.
    Urutkan tangan berdasarkan wrist.x supaya konsisten.
    """
    feat = np.zeros(FEAT_DIM, dtype=np.float32)
    if not results.multi_hand_landmarks:
        return feat

    hands = list(results.multi_hand_landmarks)
    hands_sorted = sorted(hands, key=lambda h: h.landmark[0].x)[:MAX_HANDS]

    for i, hand in enumerate(hands_sorted):
        feat[i * PER_HAND:(i + 1) * PER_HAND] = normalize_hand(hand.landmark)
    return feat


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main(args):
    label      = args.label
    n_samples  = args.samples
    out_dir    = os.path.join("data", label)
    os.makedirs(out_dir, exist_ok=True)

    # hitung sample yang sudah ada
    existing = sorted([f for f in os.listdir(out_dir) if f.endswith(".npy")])
    start_idx = len(existing)
    print(f"\n📂 Label: '{label}' | Target: {n_samples} samples | Sudah ada: {start_idx}")

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    hands_detector = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    sample_count = start_idx

    try:
        while sample_count < start_idx + n_samples:
            # ── COUNTDOWN sebelum rekam ───────────────────────────────────────
            countdown_end = time.time() + 3
            print(f"\n⏳ Sample {sample_count + 1}/{start_idx + n_samples} – bersiap...")
            while time.time() < countdown_end:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                remaining = int(countdown_end - time.time()) + 1
                cv2.putText(frame, f"Label: {label}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
                cv2.putText(frame, f"Bersiap... {remaining}",
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 200, 255), 3)
                cv2.putText(frame, f"Sample {sample_count + 1} / {start_idx + n_samples}",
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                cv2.imshow("Collect Data – tekan Q untuk berhenti", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # ── REKAM 3 detik ─────────────────────────────────────────────────
            frames_data = []
            rec_start   = time.time()
            print(f"   🔴 Rekam SEKARANG...")

            while time.time() - rec_start < RECORD_SECONDS:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res   = hands_detector.process(rgb)
                feat  = frame_to_feature(res)
                frames_data.append(feat)

                # gambar landmark
                if res.multi_hand_landmarks:
                    for hand_lm in res.multi_hand_landmarks:
                        mp_draw.draw_landmarks(
                            frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                elapsed  = time.time() - rec_start
                progress = int((elapsed / RECORD_SECONDS) * 30)
                bar      = "█" * progress + "░" * (30 - progress)
                cv2.putText(frame, f"REKAM: [{bar}]",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(frame, f"{RECORD_SECONDS - elapsed:.1f}s tersisa",
                            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.imshow("Collect Data – tekan Q untuk berhenti", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt

            # ── SIMPAN ────────────────────────────────────────────────────────
            arr = np.array(frames_data, dtype=np.float32)   # (N_frames, 126)
            if arr.shape[0] == 0:
                print("   ⚠️  Tidak ada frame – dilewati.")
                continue

            # sub-sample supaya semua sample punya panjang sama
            n_target = int(RECORD_SECONDS * TARGET_FPS)
            if len(arr) > n_target:
                idxs = np.linspace(0, len(arr) - 1, n_target, dtype=int)
                arr  = arr[idxs]
            elif len(arr) < n_target:
                # pad dengan frame terakhir
                pad = n_target - len(arr)
                arr = np.vstack([arr, np.tile(arr[-1], (pad, 1))])

            fname = os.path.join(out_dir, f"sample_{sample_count:03d}.npy")
            np.save(fname, arr)
            sample_count += 1
            print(f"   ✅ Disimpan: {fname}  shape={arr.shape}")

    except KeyboardInterrupt:
        print("\n🛑 Dihentikan oleh user.")

    finally:
        cap.release()
        hands_detector.close()
        cv2.destroyAllWindows()
        print(f"\n✔ Total sample tersimpan untuk '{label}': {sample_count}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label",   "-l", required=True, help="nama gesture, mis. halo")
    ap.add_argument("--samples", "-s", type=int, default=30, help="jumlah sample per kelas")
    ap.add_argument("--camera",  "-c", type=int, default=0,  help="index kamera (0=default)")
    main(ap.parse_args())