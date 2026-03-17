import h5py
import json

try:
    with h5py.File('object_detection.h5', 'r') as f:
        print("Keras version:", f.attrs.get('keras_version', 'Unknown'))
        print("Backend:", f.attrs.get('backend', 'Unknown'))
        if 'model_config' in f.attrs:
            config = json.loads(f.attrs['model_config'])
            print("Model config class:", config.get('class_name'))
            print("Model loaded config successfully.")
        else:
            print("No model_config found.")
except Exception as e:
    print("Error reading h5 file:", e)
