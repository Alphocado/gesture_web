// test_load.js
const tf = require("@tensorflow/tfjs-node");

async function test() {
  try {
    const model = await tf.loadLayersModel(
      "file://models/tfjs_model/model.json",
    );
    console.log("✅ Model berhasil dimuat!");
    model.summary();
  } catch (e) {
    console.error("❌ Gagal memuat model:", e.message);
  }
}

test();
