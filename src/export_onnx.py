import tensorflow as tf
import tf2onnx

MODEL_PATH = "../models/opti_fab_model_clean.keras"
ONNX_PATH = "../models/opti_fab_model.onnx"

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

spec = (tf.TensorSpec((None, 128, 128, 1), tf.float32, name="input"),)

tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=13,
    output_path=ONNX_PATH
)

print("✅ ONNX export successful")
print(f"📦 Saved at: {ONNX_PATH}")
