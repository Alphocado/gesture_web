// ==========================================
// KONFIGURASI - Sesuaikan jika perlu
// ==========================================
const MODEL_PATH = "models/tfjs_model/model.json";
const CLASSES_PATH = "models/classes.txt";
const NORM_PARAMS_PATH = "models/norm_params.json";
const VIDEO_FOLDER = "videos/";

const BUFFER_SECONDS = 3; // Durasi buffer landmark (harus sama dengan record_data.py)
const FPS = 30; // Frame per detik estimasi
const BUFFER_SIZE = BUFFER_SECONDS * FPS; // 90 frame
const CONFIDENCE_THRESHOLD = 0.7; // Minimal confidence untuk trigger video

// ==========================================
// VARIABEL GLOBAL
// ==========================================
let model = null;
let classes = [];
let normMean = null;
let normStd = null;
let handsDetector = null;
let canvasCtx = null;
let bufferLandmarks = [];
let isVideoPlaying = false;
let lastPredictionTime = 0;
const PREDICTION_COOLDOWN_MS = 1500; // Jeda antar prediksi

// ==========================================
// ELEMEN DOM
// ==========================================
const webcamEl = document.getElementById("webcam");
const canvasEl = document.getElementById("output");
const statusEl = document.getElementById("status");
const predictionSpan = document.getElementById("prediction");
const classesListSpan = document.getElementById("classes-list");
const resultVideo = document.getElementById("result-video");
const confidenceBar = document.getElementById("confidence-bar");
const confidenceFill = document.getElementById("confidence-fill");

canvasCtx = canvasEl.getContext("2d");

// ==========================================
// MUAT MODEL & METADATA
// ==========================================
async function loadModelAndMetadata() {
  try {
    statusEl.textContent = "Memuat model...";

    // Gunakan loadLayersModel (bukan loadGraphModel)
    // Sesuai dengan train_model.py yang menyimpan format tfjs_layers_model
    model = await tf.loadLayersModel(MODEL_PATH);
    console.log("✅ Model LayersModel loaded");
    console.log("   Input shape:", model.inputs[0].shape);
    console.log("   Output shape:", model.outputs[0].shape);

    // Muat daftar kelas
    const classesRes = await fetch(CLASSES_PATH);
    if (!classesRes.ok) throw new Error(`Gagal memuat ${CLASSES_PATH}`);
    const classesText = await classesRes.text();
    classes = classesText
      .trim()
      .split("\n")
      .map((c) => c.trim())
      .filter(Boolean);
    classesListSpan.textContent = classes.join(", ");
    console.log("🏷️ Classes:", classes);

    // Muat parameter normalisasi
    await loadNormParams();

    // Warm-up model dengan tensor dummy agar prediksi pertama tidak lambat
    const warmupInput = tf.zeros([1, 252]);
    const warmupOut = model.predict(warmupInput);
    warmupOut.dispose();
    warmupInput.dispose();
    console.log("🔥 Model warm-up selesai");

    statusEl.textContent = "Model siap. Meminta akses kamera...";
    return true;
  } catch (error) {
    console.error("❌ Gagal memuat model:", error);
    statusEl.textContent = `❌ Error memuat model: ${error.message}. Pastikan folder models/ sudah ada.`;
    return false;
  }
}

async function loadNormParams() {
  try {
    const res = await fetch(NORM_PARAMS_PATH);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const params = await res.json();
    normMean = tf.tensor1d(params.mean);
    normStd = tf.tensor1d(params.std);
    console.log("📊 Normalization params loaded");
  } catch (e) {
    console.warn(
      "⚠️ norm_params.json tidak ditemukan. Lanjut tanpa normalisasi:",
      e.message,
    );
    normMean = null;
    normStd = null;
  }
}

// ==========================================
// SETUP MEDIAPIPE HANDS
// ==========================================
function initHands() {
  handsDetector = new Hands({
    locateFile: (file) =>
      `https://cdn.jsdelivr.net/npm/@mediapipe/hands@0.4.1646424915/${file}`,
  });

  handsDetector.setOptions({
    maxNumHands: 2,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });

  handsDetector.onResults(onHandsResults);
}

// ==========================================
// EKSTRAK LANDMARK KONSISTEN (Kiri dulu, Kanan kedua)
// Logika ini HARUS IDENTIK dengan record_data.py
// ==========================================
function extractLandmarksOrdered(results) {
  let leftHand = null;
  let rightHand = null;

  if (results.multiHandLandmarks && results.multiHandedness) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const label = results.multiHandedness[i].label; // "Left" atau "Right"
      if (label === "Left") {
        leftHand = results.multiHandLandmarks[i];
      } else {
        rightHand = results.multiHandLandmarks[i];
      }
    }
  }

  const landmarks = [];
  for (const hand of [leftHand, rightHand]) {
    if (hand !== null) {
      for (const lm of hand) {
        landmarks.push(lm.x, lm.y, lm.z);
      }
    } else {
      // Tangan tidak terdeteksi → isi 0
      for (let j = 0; j < 63; j++) landmarks.push(0.0);
    }
  }

  return landmarks; // 126 nilai
}

// ==========================================
// CALLBACK SETIAP FRAME MEDIAPIPE
// ==========================================
function onHandsResults(results) {
  // Gambar ke canvas
  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  canvasCtx.drawImage(results.image, 0, 0, canvasEl.width, canvasEl.height);

  if (results.multiHandLandmarks) {
    for (let i = 0; i < results.multiHandLandmarks.length; i++) {
      const landmarks = results.multiHandLandmarks[i];
      const isLeft = results.multiHandedness[i]?.label === "Left";

      // Warna berbeda untuk tangan kiri/kanan
      drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {
        color: isLeft ? "#00FF00" : "#FF6600",
        lineWidth: 2,
      });
      drawLandmarks(canvasCtx, landmarks, {
        color: isLeft ? "#FF0000" : "#FF8800",
        lineWidth: 1,
        radius: 3,
      });
    }
  }
  canvasCtx.restore();

  // Jangan proses jika video sedang main
  if (isVideoPlaying) return;

  // Ekstrak & buffer landmark
  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    const landmarks = extractLandmarksOrdered(results);
    bufferLandmarks.push(landmarks);
    if (bufferLandmarks.length > BUFFER_SIZE) {
      bufferLandmarks.shift();
    }
  }

  // Update UI buffer
  const bufLen = bufferLandmarks.length;
  const pct = Math.round((bufLen / BUFFER_SIZE) * 100);
  statusEl.textContent = `Buffer: ${bufLen}/${BUFFER_SIZE} (${pct}%)`;

  if (bufLen === BUFFER_SIZE) {
    const now = Date.now();
    if (now - lastPredictionTime > PREDICTION_COOLDOWN_MS) {
      lastPredictionTime = now;
      predictGesture([...bufferLandmarks]); // copy agar aman
    }
  }
}

// ==========================================
// PREDIKSI GESTURE
// ==========================================
async function predictGesture(landmarkSequence) {
  if (!model || landmarkSequence.length === 0) return;

  // Gunakan tf.tidy agar semua tensor intermediate otomatis dibersihkan
  const scores = tf.tidy(() => {
    // 1. Buat tensor 2D [n_frames, 126]
    const seqTensor = tf.tensor2d(landmarkSequence);

    // 2. Hitung mean dan variance sepanjang axis 0 (antar frame)
    const moments = tf.moments(seqTensor, 0);
    const mean = moments.mean; // shape [126]
    const variance = moments.variance; // shape [126]
    const std = variance.sqrt(); // shape [126] = standard deviation

    // 3. Gabung jadi vektor fitur [252]
    let features = mean.concat(std);

    // 4. Normalisasi (jika ada normMean dan normStd)
    if (normMean && normStd) {
      features = features.sub(normMean).div(normStd.add(1e-8));
    }

    // 5. Bentuk input [1, 252] dan prediksi
    const input = features.reshape([1, 252]);
    const output = model.predict(input); // output shape [1, num_classes]

    // 6. Kembalikan nilai scores (Float32Array)
    return output.dataSync();
  });

  try {
    // Cari kelas dengan score tertinggi
    let maxIdx = 0;
    let maxScore = scores[0];
    for (let i = 1; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        maxIdx = i;
      }
    }

    const predictedClass = classes[maxIdx] || `kelas_${maxIdx}`;
    const confidence = maxScore;

    // Update UI prediksi
    predictionSpan.textContent = `${predictedClass} (${(confidence * 100).toFixed(1)}%)`;

    // Update confidence bar
    if (confidenceFill) {
      confidenceFill.style.width = `${(confidence * 100).toFixed(1)}%`;
      confidenceFill.style.backgroundColor =
        confidence > CONFIDENCE_THRESHOLD ? "#22c55e" : "#f59e0b";
    }

    console.log(
      `🔍 Prediksi: ${predictedClass} | Confidence: ${(confidence * 100).toFixed(1)}%`,
    );
    console.log(
      "   Semua scores:",
      scores
        .map((s, i) => `${classes[i] || i}: ${(s * 100).toFixed(1)}%`)
        .join(", "),
    );

    if (confidence > CONFIDENCE_THRESHOLD) {
      triggerVideo(predictedClass);
      bufferLandmarks = []; // Reset buffer setelah trigger
    }
  } catch (err) {
    console.error("❌ Error saat prediksi:", err);
  }
}

// ==========================================
// TRIGGER VIDEO
// ==========================================
function triggerVideo(className) {
  if (isVideoPlaying) return;

  const videoPath = `${VIDEO_FOLDER}${className}.mp4`;
  console.log(`🎬 Memutar video: ${videoPath}`);

  resultVideo.src = videoPath;
  resultVideo.style.display = "block";
  resultVideo.style.opacity = "1";
  isVideoPlaying = true;
  statusEl.textContent = `🎬 Memutar video: ${className}`;
  predictionSpan.textContent = `✅ ${className} — Memutar video...`;

  resultVideo.play().catch((e) => {
    console.error("Gagal memutar video:", e);
    statusEl.textContent = `⚠️ Video '${className}.mp4' tidak ditemukan di folder videos/.`;
    predictionSpan.textContent = `Gesture: ${className} (video tidak ada)`;
    isVideoPlaying = false;
    resultVideo.style.display = "none";
  });

  resultVideo.onended = () => {
    isVideoPlaying = false;
    resultVideo.style.display = "none";
    statusEl.textContent = "✅ Siap. Lakukan gerakan selama 3 detik.";
    predictionSpan.textContent = "-";
    bufferLandmarks = [];
    if (confidenceFill) {
      confidenceFill.style.width = "0%";
    }
  };
}

// ==========================================
// SETUP KAMERA
// ==========================================
async function setupCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    webcamEl.srcObject = stream;

    return new Promise((resolve) => {
      webcamEl.onloadedmetadata = () => {
        webcamEl.play();
        initHands();

        async function processFrame() {
          if (webcamEl.readyState >= 2) {
            await handsDetector.send({ image: webcamEl });
          }
          requestAnimationFrame(processFrame);
        }

        processFrame();
        statusEl.textContent =
          "📸 Kamera aktif. Lakukan gerakan selama 3 detik.";
        resolve(true);
      };
    });
  } catch (err) {
    console.error("Gagal akses kamera:", err);
    if (err.name === "NotAllowedError") {
      statusEl.textContent =
        "❌ Izin kamera ditolak. Aktifkan izin kamera di browser.";
    } else {
      statusEl.textContent = `❌ Kamera error: ${err.message}`;
    }
    return false;
  }
}

// ==========================================
// INISIALISASI UTAMA
// ==========================================
(async function main() {
  console.log("🚀 Inisialisasi aplikasi...");

  const modelLoaded = await loadModelAndMetadata();
  if (!modelLoaded) return;

  const cameraReady = await setupCamera();
  if (!cameraReady) return;

  console.log("✅ Aplikasi siap.");
})();
