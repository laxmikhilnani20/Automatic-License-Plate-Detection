# Root Level Files Documentation

## Overview

The root directory contains essential configuration files, utilities, and the main Jupyter notebook that serves as the project's primary entry point.

---

## 1. automatic-number-plate-recognition-88fa4f-2.ipynb

### Purpose
Complete Jupyter notebook implementing the entire Automatic Number Plate Recognition pipeline from data loading through model training and inference.

### Type
Jupyter Notebook (.ipynb)

### Content Structure (101 cells total)

#### Section 1: Introduction & Setup (Cells 1-10)
- Project overview and objectives
- Library imports (TensorFlow, OpenCV, Pandas, NumPy)
- Configuration and path setup
- Data path definition

#### Section 2: Data Loading & Exploration (Cells 11-25)
- Load labels from CSV
- Explore dataset statistics
- Visualize sample annotations
- Check bounding box distributions

#### Section 3: Data Preprocessing (Cells 26-40)
- Convert PASCAL VOC format to YOLO format
- Normalize bounding box coordinates
- Calculate box statistics
- Validate preprocessed data

#### Section 4: Dataset Splitting (Cells 41-50)
- Split data into train/validation/test sets
- Save YOLO format files
- Create train/test directories
- Verify split distribution

#### Section 5: Model Architecture (Cells 51-65)
- Define YOLO model architecture
- Load pre-trained weights
- Configure model parameters
- Summary of model layers

#### Section 6: Training (Cells 66-80)
- Prepare training data pipeline
- Configure training parameters
- Train the model with callbacks
- Monitor training metrics

#### Section 7: Evaluation (Cells 81-95)
- Evaluate on test set
- Calculate precision/recall/mAP
- Confusion matrix analysis
- Performance visualization

#### Section 8: Inference & Detection (Cells 96-101)
- Load trained model
- Run inference on test images
- Draw bounding boxes
- Display predictions with confidence scores

### Key Libraries Used
```python
import tensorflow as tf              # Deep learning
import cv2                          # Computer vision
import pandas as pd                 # Data manipulation
import numpy as np                  # Numerical computing
import matplotlib.pyplot as plt     # Visualization
import seaborn as sns               # Statistical visualization
```

### Main Functions

**Data Loading:**
```python
def load_annotations(csv_path)
    # Loads bounding box annotations from CSV
```

**Format Conversion:**
```python
def pascal_to_yolo(xmin, ymin, xmax, ymax, width, height)
    # Converts PASCAL VOC format to YOLO normalized format
```

**Model Training:**
```python
def train_model(model, train_data, val_data, epochs=100)
    # Trains YOLO model with early stopping
```

**Inference:**
```python
def detect_license_plates(image, model)
    # Runs inference and returns detections with confidence
```

### Model Details
- **Architecture:** YOLO neural network
- **Input Size:** 640x640 pixels
- **Output:** Bounding boxes with confidence scores
- **Framework:** TensorFlow/Keras 2.11.0 compatible

### Data Formats

**Input:**
- Image files (JPEG)
- CSV with bounding box labels
- Format: [filepath, xmin, xmax, ymin, ymax]

**Processing:**
- Normalized YOLO format
- Format: [class_id, center_x_norm, center_y_norm, width_norm, height_norm]

### Execution Status
- **Status:** Not executed in current state
- **Requirements:** All dependencies installed
- **Expected Output:** Trained model saved as .h5

### Key Outputs
1. Processed `data_images/` with train/test splits
2. Trained model (if executed)
3. Predictions on test images
4. Performance metrics and visualizations
5. Detection visualizations with bounding boxes

---

## 2. requirements.txt

### Purpose
Specifies all Python package dependencies for the project.

### Location
Root directory

### Content

```
Flask                       # Web framework for APIs
tensorflow==2.11.0          # Deep learning framework
opencv-python-headless      # Computer vision (no GUI)
pytesseract                 # OCR text extraction
numpy                       # Numerical computing
Werkzeug                    # Flask WSGI utilities
Jinja2                      # Template engine
MarkupSafe                  # String escaping utilities
```

### Package Descriptions

| Package | Version | Purpose |
|---------|---------|---------|
| Flask | Latest | Web API and UI framework |
| TensorFlow | 2.11.0 | Deep learning framework |
| OpenCV | Headless | Image processing (no display) |
| Pytesseract | Latest | OCR interface |
| NumPy | Latest | Numerical operations |
| Werkzeug | Latest | WSGI application utilities |
| Jinja2 | Latest | Template rendering |
| MarkupSafe | Latest | Safe HTML class generation |

### Installation

```bash
# Install all dependencies
pip install -r requirements.txt

# Or individually
pip install Flask tensorflow==2.11.0 opencv-python-headless pytesseract numpy
```

### Version Specification

**Pinned Version:**
- `tensorflow==2.11.0` - Specific TensorFlow version for model compatibility

**Flexible Versions:**
- All others without version numbers - Uses latest compatible

### Compatibility

**Python Version:** 3.8+
**Operating Systems:** macOS, Linux, Windows
**GPU Support:** Optional with CUDA toolkit

### Additional Dependencies (Not in requirements.txt)

These may be needed:
```bash
pip install scikit-learn      # For train/test splitting
pip install pandas            # For data manipulation
pip install matplotlib        # For visualization
pip install seaborn           # For statistical plots
```

---

## 3. data.yaml

### Purpose
Configuration file for YOLO dataset describing train/validation paths and class definitions.

### Location
Root directory

### Content

```yaml
train: data_images/train
val: data_images/test
nc: 1
names: ['license_plate']
```

### Field Descriptions

| Field | Value | Description |
|-------|-------|-------------|
| `train` | data_images/train | Path to training dataset |
| `val` | data_images/test | Path to validation dataset |
| `nc` | 1 | Number of classes |
| `names` | ['license_plate'] | Class names list |

### YAML Format Explanation

```yaml
train:          # Key for training path
data_images/    # Relative path to training files
train           # Subdirectory name

val:            # Value for validation path
data_images/    # Relative path to validation files
test            # Subdirectory name

nc: 1           # Single class: license_plate
names:          # Array of class names
  - 'license_plate'  # Class ID 0
```

### Usage in YOLO Training

```python
model = YOLO('yolov5s.yaml')
results = model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640
)
```

### File Format Details

- **Type:** YAML (YAML Ain't Markup Language)
- **Encoding:** UTF-8
- **No special characters or spaces in paths

### Extension Usage

Can be extended to multiple classes:

```yaml
train: data_images/train
val: data_images/test
test: data_images/test
nc: 2
names: ['license_plate', 'vehicle_body']
```

---

## 4. labels.csv

### Purpose
Extracted bounding box annotations from XML files in tabular format.

### Location
Root directory

### Format

```
filepath,xmin,xmax,ymin,ymax
/Users/asik/Desktop/ANPR/images/N517.xml,175,290,228,255
/Users/asik/Desktop/ANPR/images/N271.xml,262,318,141,160
...
```

### Column Descriptions

| Column | Description | Type | Example |
|--------|-------------|------|---------|
| filepath | Full path to source XML file | String | `/path/to/N517.xml` |
| xmin | Left boundary of bounding box | Integer | 175 |
| xmax | Right boundary of bounding box | Integer | 290 |
| ymin | Top boundary of bounding box | Integer | 228 |
| ymax | Bottom boundary of bounding box | Integer | 255 |

### Data Interpretation

For example: `175,290,228,255`
- **x-axis range:** 175 to 290 pixels (width: 115 pixels)
- **y-axis range:** 228 to 255 pixels (height: 27 pixels)
- **Position:** License plate at coordinates (175, 228) to (290, 255)

### Statistics

- **Total Records:** 400+ annotation entries
- **Image Coverage:** Multiple annotations per image
- **Data Types:** All integer coordinates
- **Format:** PASCAL VOC bounding box format

### File Format

- **Extension:** .csv (Comma-Separated Values)
- **Delimiter:** Comma (,)
- **Header Row:** Yes (filepath,xmin,xmax,ymin,ymax)
- **Encoding:** UTF-8

### Usage Example

```python
import pandas as pd

# Load labels
df = pd.read_csv('labels.csv')

# Display first few rows
print(df.head())

# Calculate box statistics
df['width'] = df['xmax'] - df['xmin']
df['height'] = df['ymax'] - df['ymin']
print(f"Average plate width: {df['width'].mean()}")
print(f"Average plate height: {df['height'].mean()}")
```

### Related Files

- **Original Format:** XML files in `images/` directory
- **YOLO Format:** `.txt` files in `data_images/`
- **Source Data in Notebook:** Conversion shown in notebook cells 15-25

---

## 5. test_load.py

### Purpose
Utility script to test whether the pre-trained model can be loaded successfully.

### Location
Root directory

### Content

```python
import tensorflow as tf
import sys

print(f"Python version: {sys.version}")
try:
    model = tf.keras.models.load_model('./object_detection.h5')
    print("SUCCESS: Model loaded successfully.")
except Exception as e:
    print(f"FAILURE: {type(e).__name__}: {e}")
```

### Functionality

**Three Main Operations:**

1. **Print Python Version**
   ```python
   print(f"Python version: {sys.version}")
   ```
   - Displays current Python interpreter version

2. **Load Model**
   ```python
   model = tf.keras.models.load_model('./object_detection.h5')
   ```
   - Attempts to load the pre-trained model

3. **Report Status**
   - Success: Prints "SUCCESS: Model loaded successfully."
   - Failure: Prints exception type and message

### Expected Output

**Success:**
```
Python version: 3.9.x ...
SUCCESS: Model loaded successfully.
```

**Failure:**
```
Python version: 3.9.x ...
FAILURE: FileNotFoundError: [Errno 2] No such file or directory: './object_detection.h5'
```

### Use Cases

1. **Dependency Verification** - Check if TensorFlow is installed correctly
2. **Model File Validation** - Verify model file exists and is readable
3. **Environment Testing** - Confirm environment setup before running full pipeline
4. **Debugging** - Quick test when encountering issues

### Execution

```bash
python test_load.py
```

### Common Issues

| Issue | Solution |
|-------|----------|
| FileNotFoundError | Ensure `object_detection.h5` exists in current directory |
| ImportError (tf) | Install TensorFlow: `pip install tensorflow==2.11.0` |
| Model corruption | Re-download or retrain the model |

---

## 6. inspect_h5.py

### Purpose
Utility to inspect the structure and metadata of the H5 model file.

### Location
Root directory

### Content

```python
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
```

### Functionality

**Inspects:**

1. **Keras Version** - Which Keras version created the model
   ```python
   f.attrs.get('keras_version', 'Unknown')
   ```

2. **Backend** - Deep learning backend (TensorFlow, Theano, etc.)
   ```python
   f.attrs.get('backend', 'Unknown')
   ```

3. **Model Configuration** - Complete model architecture
   ```python
   config = json.loads(f.attrs['model_config'])
   ```

4. **Model Class** - Type of model (Sequential, Functional, etc.)
   ```python
   config.get('class_name')
   ```

### Expected Output

Example output from a valid H5 file:
```
Keras version: 2.6.0
Backend: tensorflow
Model config class: Model
Model loaded config successfully.
```

### Use Cases

1. **Compatibility Check** - Verify Keras/TensorFlow versions
2. **Model Architecture Inspection** - Review model layers and parameters
3. **Debugging** - Troubleshoot model loading issues
4. **Documentation** - Record model metadata

### Execution

```bash
python inspect_h5.py
```

### H5 File Structure

An H5 file contains:

```
object_detection.h5
├── attrs/                    # Metadata attributes
│   ├── keras_version
│   ├── backend
│   ├── model_config
│   └── training_config
├── model_weights/           # Learned parameters
├── model_metadata/          # Architecture info
└── layer_names/             # Named layers
```

### Common Issues

| Issue | Meaning |
|-------|---------|
| "No model_config found" | File is incomplete or corrupted |
| FileNotFoundError | H5 file not in current directory |
| "Error reading h5 file" | File is not valid H5 format |

---

## 7. README.md

### Purpose
Project overview and getting started guide in Markdown format.

### Location
Root directory

### Content Sections

#### RESULTS
- Links to demonstration GIFs
- Shows real-time plate detection examples
- Visual proof of concept

#### DETECT LICENSE PLATES WITH THE YOLO ALGORITHM
- Main project headline
- Describes algorithm choice

#### Overview
- Security importance
- Solution description
- Expected accuracy (~90-95%)

#### Dataset Information
- 453 image files
- JPEG format
- PASCAL VOC annotations
- Format description

#### Table of Content
1. Labeling, Understanding & Collecting Data
2. Data Processing
3. Deep Learning for Object Detection
4. Pipeline Object Detection Model
5. Optical Character Recognition (OCR)
6. Number Plate Web App
7. Real-time Number Plate Recognition with YOLO

### Key Information

**Project Goal:** Automatically recognize license plates using Python
**Accuracy:** 90-95% on low-resolution images
**Technologies:** OpenCV, Pytesseract, YOLO
**Output:** Web app and real-time recognition system

---

## Summary Table

| File | Type | Purpose | Size |
|------|------|---------|------|
| notebook | .ipynb | Main implementation | Large |
| requirements.txt | .txt | Dependencies | ~10 lines |
| data.yaml | .yaml | YOLO config | 5 lines |
| labels.csv | .csv | Annotations | 400+ lines |
| test_load.py | .py | Model test utility | ~10 lines |
| inspect_h5.py | .py | Model inspection | ~15 lines |
| README.md | .md | Documentation | ~50 lines |

---

## Quick Reference

### To Test Setup
```bash
python test_load.py
```

### To Inspect Model
```bash
python inspect_h5.py
```

### To Install Dependencies
```bash
pip install -r requirements.txt
```

### To Run Notebook
```bash
jupyter notebook automatic-number-plate-recognition-88fa4f-2.ipynb
```
