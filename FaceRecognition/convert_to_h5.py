import tensorflow as tf

model = tf.keras.models.load_model("models/liveness.model")
model.save("models/liveness_model.h5")

print("[INFO] Converted to HDF5 format successfully.")
