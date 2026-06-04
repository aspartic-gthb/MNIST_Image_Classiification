"""
convert_model.py
Converts best_model.keras to TensorFlow.js LayersModel format
Output: docs/model/model.json + weight shards
"""

import os, sys, json, struct, numpy as np

# ── Try importing tensorflowjs converter directly ────────────────────────────
try:
    import tensorflowjs as tfjs
    HAS_TFJS = True
except ImportError:
    HAS_TFJS = False

import tensorflow as tf

MODEL_PATH  = "best_model.keras"
OUTPUT_DIR  = os.path.join("docs", "model")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"TensorFlow version : {tf.__version__}")
print(f"tensorflowjs pkg   : {'available' if HAS_TFJS else 'NOT installed — using manual export'}")
print(f"Loading model from : {MODEL_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

# ── Path 1: use tensorflowjs package if available ───────────────────────────
if HAS_TFJS:
    tfjs.converters.save_keras_model(model, OUTPUT_DIR)
    print(f"\n✅  Model saved to '{OUTPUT_DIR}' via tensorflowjs package")
    sys.exit(0)

# ── Path 2: manual serialisation (no tensorflowjs package needed) ────────────
print("\nManual TF.js serialisation starting…")

SHARD_SIZE = 4 * 1024 * 1024   # 4 MB shards

weights_manifest = []
all_bytes = bytearray()

for layer in model.layers:
    lw = layer.get_weights()
    if not lw:
        continue
    entry = {"name": layer.name, "weights": []}
    for i, w in enumerate(lw):
        var_name   = f"{layer.name}/weight_{i}"
        byte_data  = w.astype(np.float32).tobytes()
        byte_offset= len(all_bytes)
        all_bytes += byte_data
        entry["weights"].append({
            "name":   var_name,
            "shape":  list(w.shape),
            "dtype":  "float32",
            "byteLength": len(byte_data)
        })
    weights_manifest.append(entry)

# Write binary shards
shard_files = []
for shard_i, offset in enumerate(range(0, len(all_bytes), SHARD_SIZE)):
    chunk     = all_bytes[offset:offset + SHARD_SIZE]
    shard_name= f"group1-shard{shard_i+1}of{-(-len(all_bytes)//SHARD_SIZE)}.bin"
    shard_path= os.path.join(OUTPUT_DIR, shard_name)
    with open(shard_path, "wb") as f:
        f.write(chunk)
    shard_files.append(shard_name)
    print(f"  Wrote {shard_path} ({len(chunk)/1024:.1f} KB)")

# Build model topology
topology = json.loads(model.to_json())

# Build model.json
model_json = {
    "format": "layers-model",
    "generatedBy": f"TensorFlow {tf.__version__} (manual export)",
    "convertedBy": "convert_model.py",
    "modelTopology": topology,
    "weightsManifest": [
        {
            "paths": shard_files,
            "weights": [
                {
                    "name":  w["name"],
                    "shape": w["shape"],
                    "dtype": w["dtype"]
                }
                for entry in weights_manifest
                for w in entry["weights"]
            ]
        }
    ]
}

model_json_path = os.path.join(OUTPUT_DIR, "model.json")
with open(model_json_path, "w") as f:
    json.dump(model_json, f)

print(f"\n✅  model.json written to '{model_json_path}'")
print(f"    Total weight bytes : {len(all_bytes)/1024/1024:.2f} MB")
print(f"    Shards             : {len(shard_files)}")
print(f"\nDone! Copy the '{OUTPUT_DIR}' folder into your web app.")
