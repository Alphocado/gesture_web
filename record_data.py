import cv2
import mediapipe as mp
import csv
import os
import time
import argparse
from datetime import datetime

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Konfigurasi dari argumen command line
parser = argparse.ArgumentParser(description="Rekam data landmark tangan untuk pengenalan gerakan.")
parser.add_argument("--label", type=str, required=True, help="Label gerakan (contoh: halo, tos, dll)")
parser.add_argument("--samples", type=int, default=30, help="Jumlah sampel (default: 30)")
parser.add_argument("--duration", type=float, default=3.0, help="Durasi tiap rekaman dalam detik (default: 3.0)")
args = parser.parse_args()

LABEL = args.label
TOTAL_SAMPLES = args.samples
RECORD_DURATION = args.duration

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

existing_files = [f for f in os.listdir(DATA_DIR) if f.startswith(f"{LABEL}_") and f.endswith(".csv")]
current_count = len(existing_files)
print(f"📁 Ditemukan {current_count} sampel untuk label '{LABEL}'.")
print(f"🎯 Target: {TOTAL_SAMPLES} sampel.")
if current_count >= TOTAL_SAMPLES:
    print("✅ Target sudah tercapai. Tidak perlu merekam lagi.")
    exit()

need_samples = TOTAL_SAMPLES - current_count
print(f"📽️  Akan merekam {need_samples} sampel lagi.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Tidak dapat membuka webcam.")
    exit()

recording = False
record_start_time = 0
frames_data = []
samples_recorded = 0

print("\n📌 Instruksi:")
print(f"   - Tekan SPASI untuk mulai merekam 1 sampel ({RECORD_DURATION} detik).")
print("   - Selama perekaman, lakukan gerakan secara natural.")
print("   - Setelah selesai, otomatis tersimpan.")
print("   - Tekan 'q' untuk keluar.\n")


def extract_landmarks_ordered(results):
    """
    Ekstrak landmark 2 tangan secara konsisten: kiri dulu, kanan kedua.
    Jika tangan tidak terdeteksi, isi dengan 0.
    Ini HARUS sama persis dengan logika di script.js.
    """
    left_lm = None
    right_lm = None

    if results.multi_hand_landmarks and results.multi_handedness:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            label = results.multi_handedness[i].classification[0].label
            # Perhatian: frame sudah di-flip, jadi "Left" dari MediaPipe = tangan kanan user
            # tapi kita pakai label apa adanya agar konsisten dengan JS
            if label == "Left":
                left_lm = hand_landmarks
            else:
                right_lm = hand_landmarks

    landmarks = []
    for hand in [left_lm, right_lm]:
        if hand is not None:
            for lm in hand.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        else:
            landmarks.extend([0.0] * 63)  # 21 titik * 3 koordinat

    return landmarks  # Total: 126 nilai


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = hands.process(rgb)
    rgb.flags.writeable = True

    # Gambar landmark
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

    info_text = f"Label: {LABEL} | Sampel: {samples_recorded + current_count}/{TOTAL_SAMPLES}"
    cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    if recording:
        elapsed = time.time() - record_start_time
        remaining = RECORD_DURATION - elapsed
        cv2.putText(frame, f"● REC {remaining:.1f}s", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if elapsed < RECORD_DURATION:
            if results.multi_hand_landmarks:
                lm = extract_landmarks_ordered(results)
                frames_data.append(lm)
        else:
            # Rekaman selesai
            recording = False
            if len(frames_data) >= 5:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = f"{LABEL}_{timestamp}.csv"
                filepath = os.path.join(DATA_DIR, filename)
                with open(filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['label'] + [f'lm{i}' for i in range(126)]
                    writer.writerow(header)
                    for frame_landmarks in frames_data:
                        writer.writerow([LABEL] + frame_landmarks)
                samples_recorded += 1
                total = samples_recorded + current_count
                print(f"✅ Sampel {total}/{TOTAL_SAMPLES} disimpan: {filename} ({len(frames_data)} frame)")
            else:
                print(f"⚠️ Hanya {len(frames_data)} frame terdeteksi. Coba lagi dengan tangan lebih jelas.")

            frames_data = []
            if samples_recorded + current_count >= TOTAL_SAMPLES:
                print(f"\n🎉 Target {TOTAL_SAMPLES} sampel untuk label '{LABEL}' tercapai!")
                break

    cv2.imshow('Hand Gesture Recorder', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and not recording:
        recording = True
        record_start_time = time.time()
        frames_data = []
        print(f"⏺️  Merekam sampel ke-{samples_recorded + current_count + 1}...")
    elif key == ord('q'):
        print("⛔ Keluar dari program.")
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n📁 Selesai. Total baru direkam: {samples_recorded} sampel. Data tersimpan di folder 'data/'.")