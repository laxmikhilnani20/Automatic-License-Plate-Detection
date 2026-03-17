import tensorflow as tf
import sys

print(f"Python version: {sys.version}")
try:
    model = tf.keras.models.load_model('./object_detection.h5')
    print("SUCCESS: Model loaded successfully.")
except Exception as e:
    print(f"FAILURE: {type(e).__name__}: {e}")
