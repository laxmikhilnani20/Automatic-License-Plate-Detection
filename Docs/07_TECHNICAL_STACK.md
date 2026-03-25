# Technical Stack Documentation

## Complete Technology Overview

This document provides detailed information about all technologies, frameworks, libraries, and tools used in the Automatic License Plate Detection project.

---

## Technology Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
│  Jupyter Notebook | Flask Web App | Command Line CLI       │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Framework Layer                            │
│  TensorFlow/Keras | PyTorch/YOLO | Scikit-learn           │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Libraries & Tools Layer                       │
│  OpenCV | Numpy | Pandas | Tesseract | ONNX               │
└────────────────────────┬────────────────────────────────────┘
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Hardware/OS Layer                         │
│  CPU/GPU | macOS/Linux/Windows | Memory/Storage            │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Dependencies

### 1. TensorFlow/Keras

**Purpose:** Deep learning framework for model training
**Version:** 2.11.0 (pinned)
**Type:** Core framework

**Installation:**
```bash
pip install tensorflow==2.11.0
```

**Key Components:**
- **keras.models** - Model building
- **keras.layers** - Neural network layers
- **keras.optimizers** - Training optimizers
- **keras.callbacks** - Training callbacks
- **keras.utils** - Utility functions

**Usage in Project:**
```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Load model
model = tf.keras.models.load_model('object_detection.h5')

# Build model
model = models.Sequential([
    layers.Conv2D(64, 3, activation='relu', input_shape=(640, 640, 3)),
    layers.MaxPooling2D(2),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Train
model.fit(train_data, epochs=100)
```

**Why TensorFlow 2.11.0:**
- Stable version
- Good performance
- Supporting libraries well-tested
- Compatible with Keras API

### 2. YOLOv5

**Purpose:** State-of-the-art object detection architecture
**Provider:** Ultralytics
**Repository:** https://github.com/ultralytics/yolov5

**Installation:**
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

**Model Variants:**
- **yolov5n** - Nano (fastest, smallest)
- **yolov5s** - Small ← Used in this project
- **yolov5m** - Medium (balanced)
- **yolov5l** - Large (slower, more accurate)
- **yolov5x** - Extra Large (most accurate)

**Key Features:**
- Real-time detection
- High accuracy (>95%)
- Efficient inference
- Easy to train on custom data
- Supports multiple export formats

**Training Command:**
```bash
python train.py --data data.yaml --weights yolov5s.pt --epochs 100 --img 640 --batch-size 8
```

**Inference:**
```python
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
results = model(images)
detections = results.xyxy[0]  # Get detections
```

---

## Computer Vision Libraries

### 3. OpenCV (opencv-python-headless)

**Purpose:** Computer vision and image processing
**Type:** Optional but recommended
**Package:** `opencv-python-headless` (no GUI)

**Installation:**
```bash
pip install opencv-python-headless
```

**Key Modules:**
- **cv2.imread()** - Read images
- **cv2.imwrite()** - Write images
- **cv2.resize()** - Resize images
- **cv2.cvtColor()** - Color space conversion
- **cv2.rectangle()** - Draw rectangles
- **cv2.putText()** - Draw text

**Usage Examples:**

```python
import cv2

# Read image
image = cv2.imread('image.jpg')

# Resize
resized = cv2.resize(image, (640, 640))

# Convert color space
gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Draw bounding box
cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Write text
cv2.putText(image, 'Label', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save
cv2.imwrite('output.jpg', image)
```

**Why Headless Version:**
- No GUI dependencies
- Smaller package size
- Better for server environments
- No display needed for batch processing

---

## Data Processing Libraries

### 4. Pandas

**Purpose:** Data manipulation and analysis
**Type:** Data processing
**Typical Version:** Latest

**Installation:**
```bash
pip install pandas
```

**Key Features:**
```python
import pandas as pd

# Read CSV
df = pd.read_csv('labels.csv')

# Data inspection
print(df.head())
print(df.describe())
print(df.info())

# Data filtering
filtered = df[df['width'] > 100]

# Data transformation
df['area'] = df['width'] * df['height']

# Groupby operations
stats = df.groupby('class')['area'].mean()

# Save to CSV
df.to_csv('processed.csv', index=False)
```

**Usage in Project:**
- Load annotations CSV
- Data validation
- Statistics calculation
- DataFrame operations for preprocessing

### 5. NumPy

**Purpose:** Numerical computing and arrays
**Type:** Numerical library
**Typical Version:** Latest

**Installation:**
```bash
pip install numpy
```

**Key Operations:**
```python
import numpy as np

# Arrays
arr = np.array([[1, 2], [3, 4]])
arr = np.zeros((10, 10))
arr = np.ones((5, 5))

# Array operations
result = np.mean(arr, axis=0)
result = np.std(arr)

# Reshaping
reshaped = arr.reshape((20, 2))

# Element-wise operations
result = arr * 2
result = arr + 10

# Image operations
image = np.random.rand(640, 640, 3)  # Random image
normalized = image / 255.0  # Normalize
```

**Usage in Project:**
- Array operations on images
- Numerical calculations
- Matrix operations for preprocessing
- Integration with TensorFlow/Keras

---

## OCR (Optical Character Recognition)

### 6. Pytesseract

**Purpose:** Python interface to Tesseract OCR engine
**Type:** OCR interface
**Package:** pytesseract

**Installation:**
```bash
pip install pytesseract
```

**System Dependency - Tesseract Engine:**

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr
```

**Windows:**
- Download from: https://github.com/UB-Mannheim/tesseract/wiki

**Verify Installation:**
```bash
tesseract --version
```

**Usage:**
```python
import pytesseract
from PIL import Image

# Load image
image = Image.open('license_plate.jpg')

# Extract text
text = pytesseract.image_to_string(image)
print(f"Extracted: {text}")

# With configuration
text = pytesseract.image_to_string(
    image,
    config='--psm 8'  # Single character mode
)

# Get detailed results
data = pytesseract.image_to_data(image, output_type='dict')
print(data['text'])      # Text
print(data['conf'])      # Confidence
```

**PSM Modes:**
```
0 = Orientation and script detection
3 = Fully automatic (DEFAULT)
4 = Single column
6 = Single block
7 = Single text line
8 = Single word
11 = Sparse text
```

### 7. Pillow (PIL)

**Purpose:** Image processing and manipulation
**Type:** Image library
**Typical Version:** Latest

**Installation:**
```bash
pip install Pillow
```

**Usage:**
```python
from PIL import Image, ImageEnhance, ImageFilter

# Open image
image = Image.open('image.jpg')

# Convert
gray = image.convert('L')
rgb = image.convert('RGB')

# Enhance contrast
enhancer = ImageEnhance.Contrast(image)
enhanced = enhancer.enhance(2.0)

# Apply filter
filtered = image.filter(ImageFilter.MedianFilter(size=3))

# Resize
resized = image.resize((640, 640))

# Save
image.save('output.jpg')
```

**Integration with OpenCV:**
```python
import cv2
from PIL import Image
import numpy as np

# Convert OpenCV to PIL
gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pil_image = Image.fromarray(gray_cv)

# Convert PIL to OpenCV
np_array = np.array(pil_image)
cv_image = cv2.cvtColor(np_array, cv2.COLOR_RGB2BGR)
```

---

## Web Framework & Utilities

### 8. Flask

**Purpose:** Web framework for API and UI
**Type:** Web framework
**Typical Version:** Latest

**Installation:**
```bash
pip install Flask
```

**Application Structure:**
```python
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/api/detect', methods=['POST'])
def detect_plate():
    """API endpoint for plate detection"""
    file = request.files['image']
    
    # Process image
    detections = model.predict(file)
    
    return jsonify({
        'detections': detections,
        'success': True
    })

@app.route('/')
def index():
    """Serve UI"""
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Related Packages:**
- **Werkzeug** - WSGI utilities
- **Jinja2** - Template engine
- **MarkupSafe** - Safe string escaping

---

## Machine Learning Utilities

### 9. Scikit-learn

**Purpose:** Machine learning utilities
**Type:** ML library
**Typical Version:** Latest

**Installation:**
```bash
pip install scikit-learn
```

**Common Usage:**
```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Calculate metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
```

---

## Model Export Formats

### 10. ONNX (Open Neural Network Exchange)

**Purpose:** Framework-agnostic model format
**Package:** onnx, onnxruntime

**Installation:**
```bash
pip install onnx onnxruntime
```

**Export:**
```python
import torch
import onnx

model = torch.load('best.pt')
dummy_input = torch.randn(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=12,
    input_names=['images'],
    output_names=['output']
)
```

**Inference:**
```python
import onnxruntime as rt

session = rt.InferenceSession('model.onnx')
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

result = session.run([output_name], {input_name: input_data})
```

### 11. PyTorch

**Purpose:** Deep learning framework (used by YOLO)
**Type:** Framework
**Typical Version:** Latest

**Installation:**
```bash
pip install torch torchvision torchaudio
```

**Usage with YOLO:**
```python
import torch
import torchvision

# Load YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Inference
results = model(images)

# Save model
torch.save(model.state_dict(), 'model.pt')

# Load model
model.load_state_dict(torch.load('model.pt'))
```

---

## Visualization Libraries

### 12. Matplotlib

**Purpose:** 2D plotting and visualization
**Type:** Visualization
**Installation:**
```bash
pip install matplotlib
```

**Common Plots:**
```python
import matplotlib.pyplot as plt

# Line plot
plt.plot([1, 2, 3], [1, 4, 9])
plt.title('Title')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Histogram
plt.hist(data, bins=50)

# Image display
plt.imshow(image)
plt.axis('off')

# Multiple subplots
fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(image1)
axes[0, 1].imshow(image2)
```

### 13. Seaborn

**Purpose:** Statistical data visualization
**Type:** Visualization
**Installation:**
```bash
pip install seaborn
```

**Common Visualizations:**
```python
import seaborn as sns

# Heatmap (for confusion matrix)
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')

# Distribution plot
sns.histplot(data)

# Correlation matrix
sns.heatmap(correlation_matrix, annot=True)
```

---

## Development & Documentation Tools

### 14. Jupyter

**Purpose:** Interactive notebook environment
**Packages:** jupyter, jupyterlab
**Installation:**
```bash
pip install jupyter jupyterlab
```

**Launch:**
```bash
jupyter notebook
jupyter lab
```

---

## Dependency Management

### Package Manager

**pip** - Python package installer (included with Python 3.8+)

**Install all dependencies:**
```bash
pip install -r requirements.txt
```

**Requirements File Format:**
```
Flask
tensorflow==2.11.0
opencv-python-headless
pytesseract
numpy
Werkzeug
Jinja2
MarkupSafe
```

**Version Specifications:**
- `package` - Latest version
- `package==2.0.0` - Exact version (pinned)
- `package>=2.0.0` - Minimum version
- `package>=2.0.0,<3.0.0` - Version range

---

## Hardware Requirements

### CPU

**Minimum:**
- Intel i5 / AMD Ryzen 5
- 4 cores, 3-4 GHz
- Training time: ~5-10 hours
- Inference: ~100ms per image

**Recommended:**
- Intel i7+ / AMD Ryzen 7+
- 8+ cores
- Training time: ~2-5 hours
- Inference: ~50ms per image

### GPU (Optional but Recommended)

**NVIDIA GPU:**
- Compute Capability: 3.5 or higher
- VRAM: 4GB minimum (8GB+ recommended)
- CUDA Toolkit: 11.8+
- cuDNN: 8.0+
- Training: ~30-60 minutes
- Inference: ~10-20ms per image

**Installation:**
```bash
# CUDA Toolkit
# cuDNN
# Then install TensorFlow GPU version
pip install tensorflow[and-cuda]
```

### RAM

- **Minimum:** 8GB
- **Recommended:** 16GB
- **For batch processing:** 32GB+

### Storage

- **Code & data:** ~5-10GB
- **Models:** ~200-500MB
- **Results:** Variable (1GB+)
- **Total:** ~10-20GB recommended

---

## Operating System Support

### macOS
- ✅ Fully supported
- CPU training available
- GPU support limited (Metal Performance Shaders)

### Linux (Ubuntu 20.04+)
- ✅ Fully supported
- CPU and GPU training
- Recommended for production

### Windows 10/11
- ✅ Supported
- CPU and GPU training
- WSL2 recommended for better compatibility

---

## Version Compatibility Matrix

| Component | Version | Python | TensorFlow | CUDA |
|-----------|---------|--------|------------|------|
| TensorFlow | 2.11.0 | 3.8-3.10 | 2.11.0 | 11.8+ |
| PyTorch | Latest | 3.8+ | - | 11.7+ |
| OpenCV | Latest | 3.7+ | - | - |
| Pandas | Latest | 3.8+ | - | - |
| NumPy | Latest | 3.8+ | - | - |

---

## Environment Setup

### Using Virtual Environment

```bash
# Create environment
python -m venv anpr_env

# Activate (macOS/Linux)
source anpr_env/bin/activate

# Activate (Windows)
anpr_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Deactivate
deactivate
```

### Using Conda

```bash
# Create environment
conda create -n anpr_env python=3.9

# Activate
conda activate anpr_env

# Install dependencies
pip install -r requirements.txt

# Deactivate
conda deactivate
```

---

## Performance Optimization

### Inference Speed

**Techniques:**
1. Model quantization (int8)
2. Model pruning (remove weights)
3. ONNX Runtime optimization
4. TensorRT conversion
5. Batch processing

**Expected speeds:**
- CPU: 100-500ms per image
- GPU: 10-50ms per image
- ONNX optimized: 5-30ms per image

### Memory Optimization

**Techniques:**
1. Model distillation (smaller model)
2. Gradient checkpointing (during training)
3. Mixed precision (float16)
4. Batch size reduction
5. Input size reduction

---

## Security Considerations

### Model Security
- Save checkpoints regularly
- Version control models
- Validate model checksums
- Use HTTPS for API

### Data Security
- Encrypt stored data
- Use secure file transfer
- Validate input data
- Sanitize file paths

### Dependency Security
- Keep packages updated
- Use virtual environments
- Check for vulnerabilities (safety package)

---

## Troubleshooting

### Common Issues

**TensorFlow import fails:**
```python
# Solution: Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.11.0
```

**CUDA not found:**
```bash
# Check NVIDIA drivers
nvidia-smi

# Install CUDA toolkit
# Reinstall TensorFlow with CUDA support
```

**Out of memory:**
- Reduce batch size
- Reduce image size
- Use GPU
- Enable mixed precision

**Slow inference:**
- Use GPU
- Use smaller model
- Batch process images
- Use ONNX Runtime

---

## Summary

**Core Technologies:**
- TensorFlow 2.11.0 - Deep learning
- YOLOv5 - Object detection
- OpenCV - Computer vision
- Python 3.8+ - Programming language

**Supporting Libraries:**
- Pandas - Data processing
- NumPy - Numerical computing
- Scikit-learn - ML utilities
- Pytesseract - OCR

**Infrastructure:**
- Flask - Web server
- Jupyter - Development notebooks
- ONNX - Model export
- PyTorch - Alternative framework

**Utilities:**
- Matplotlib - Visualization
- Seaborn - Statistical plots
- Pillow - Image processing

This comprehensive tech stack provides a robust, scalable foundation for automatic license plate recognition and can be extended for production deployments.
