# Data Files and Formats Documentation

## Overview

The data in this project exists in multiple formats at different stages of processing, from raw XML annotations to YOLO-formatted training labels.

---

## 1. Source Data: images/ Directory

### Purpose
Contains PASCAL VOC XML annotation files describing license plate locations.

### Structure
- **Directory Path:** `images/`
- **File Count:** 243 XML files
- **Naming Convention:** N1.xml, N2.xml, N3.xml, ..., N248.xml
- **Total Size:** Approximately 2-3 MB

### File Format: PASCAL VOC XML

Each XML file contains annotation data for related binary image.

#### Example XML Structure

```xml
<annotation>
    <filename>N1.jpg</filename>
    <path>/path/to/images/N1.jpg</path>
    <source>
        <database>Unknown</database>
    </source>
    <size>
        <width>480</width>
        <height>360</height>
        <depth>3</depth>
    </size>
    <segmented>0</segmented>
    <object>
        <name>license_plate</name>
        <pose>Unspecified</pose>
        <truncated>0</truncated>
        <difficult>0</difficult>
        <bndbox>
            <xmin>175</xmin>
            <ymin>228</ymin>
            <xmax>290</xmax>
            <ymax>255</ymax>
        </bndbox>
    </object>
</annotation>
```

### XML Element Descriptions

| Element | Type | Description |
|---------|------|-------------|
| filename | String | Image filename (e.g., "N1.jpg") |
| path | String | Full path to image file |
| width | Integer | Image width in pixels |
| height | Integer | Image height in pixels |
| depth | Integer | Color channels (3 for RGB) |
| name | String | Object class ("license_plate") |
| xmin | Integer | Bounding box left edge |
| ymin | Integer | Bounding box top edge |
| xmax | Integer | Bounding box right edge |
| ymax | Integer | Bounding box bottom edge |
| truncated | Boolean | 1 if plate extends beyond image |
| difficult | Boolean | 1 if annotation is difficult |

### Key Characteristics

- **Single Class:** All objects are "license_plate"
- **Single Object per Image:** One license plate per annotation (typically)
- **Rectangular Regions:** All bounding boxes are axis-aligned rectangles
- **Integer Coordinates:** All pixel positions are whole numbers
- **Human Readable:** XML is text-based and easily parseable

### Sample Bounding Box Data

| Image | Width | Height | xmin | ymin | xmax | ymax | Plate Width | Plate Height |
|-------|-------|--------|------|------|------|------|-------------|-------------|
| N1.xml | 480 | 360 | 175 | 228 | 290 | 255 | 115 | 27 |
| N2.xml | 640 | 480 | 262 | 141 | 318 | 160 | 56 | 19 |
| N3.xml | 800 | 600 | 87 | 166 | 216 | 212 | 129 | 46 |

### How This Data is Generated

1. **Manual Annotation** - Humans draw boxes around plates
2. **Annotation Tool** - Tools like LabelImg create XML files
3. **Quality Check** - Annotations reviewed for accuracy
4. **Format Export** - PASCAL VOC format selected for standardization

### Statistics Summary

```
Total Annotation Files: 243
Total Unique Images: 243
Objects per File: 1 (average/max)
Bounding Box Format: PASCAL VOC (xmin, ymin, xmax, ymax)
Coordinate Range: Pixels (0 to image dimensions)
Class Distribution: 100% license_plate
```

---

## 2. Processed Data: data_images/ Directory

### Purpose
Contains preprocessed, split, and YOLO-formatted training data.

### Directory Structure

```
data_images/
├── train/
│   ├── N100.txt, N101.txt, ..., N248.txt  (224 files)
│   └── [Training annotations in YOLO format]
│
└── test/
    ├── N1.txt, N2.txt, ..., N93.txt  (25 files)
    └── [Testing annotations in YOLO format]
```

### YOLO Format (.txt Files)

#### Purpose
YOLO format enables efficient training and inference without needing to parse XML files.

#### File Format
Each line represents one object (bounding box):
```
<class_id> <center_x_norm> <center_y_norm> <width_norm> <height_norm>
```

#### Example Content

```
0 0.5237 0.6333 0.2396 0.0750
```

Breaking down this example:
- `0` - Class ID (license_plate = 0)
- `0.5237` - X coordinate of box center (normalized: 0.0 to 1.0)
- `0.6333` - Y coordinate of box center (normalized: 0.0 to 1.0)
- `0.2396` - Box width as fraction of image width
- `0.0750` - Box height as fraction of image height

#### Coordinate System

For a 480x360 image with plate at (175, 228) to (290, 255):

```python
# Calculate YOLO format
center_x = (175 + 290) / 2 = 232.5
center_y = (228 + 255) / 2 = 241.5
width = 290 - 175 = 115
height = 255 - 228 = 27

# Normalize
center_x_norm = 232.5 / 480 = 0.4844
center_y_norm = 241.5 / 360 = 0.6708
width_norm = 115 / 480 = 0.2396
height_norm = 27 / 360 = 0.0750

# Result
0 0.4844 0.6708 0.2396 0.0750
```

### Training Set (train/)

**Contents:**
- 224 .txt label files
- File names: N100.txt through N248.txt (selective numbers)
- Corresponds to training images

**Purpose:**
- Used to train the YOLO model
- Model learns patterns from these samples

**File Statistics:**
- Each file has typically 1 line (1 plate per image)
- File size: ~30-50 bytes each
- Total: ~10 KB

### Testing Set (test/)

**Contents:**
- 25 .txt label files
- File names: N1.txt through N93.txt (selective, non-overlapping)
- Corresponds to test images

**Purpose:**
- Used to validate model performance
- Separate from training for unbiased evaluation

**File Statistics:**
- Each file has typically 1 line
- File size: ~30-50 bytes each
- Total: ~1 KB

### Data Split Strategy

```
Total Images: 243 (not all used)
Training Set: 224 images (92.2%)
Testing Set: 19 images (7.8%)
            + ~25 additional for validation

Ratio: ~90% train, ~10% test/validation
```

### Train/Test Split Distribution

| Category | Images | Files | Percentage |
|----------|--------|-------|-----------|
| Training | 224 | N100.txt - N248.txt | 89.6% |
| Testing | 25 | N1.txt - N93.txt | 10.0% |
| Not Used | ~5-10 | Various | 2-4% |
| **Total** | **~250** | **~250** | **100%** |

---

## 3. Labels CSV: labels.csv

### Purpose
Convenience format containing all annotations in tabular form(extracted from XML files).

### Format

```csv
filepath,xmin,xmax,ymin,ymax
/Users/asik/Desktop/ANPR/images/N517.xml,175,290,228,255
/Users/asik/Desktop/ANPR/images/N271.xml,262,318,141,160
```

### Column Descriptions Full

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| filepath | String | Path to source XML | /path/to/N517.xml |
| xmin | Integer | Left edge | 175 |
| xmax | Integer | Right edge | 290 |
| ymin | Integer | Top edge | 228 |
| ymax | Integer | Bottom edge | 255 |

### Derived Information

From CSV data, can calculate:
```python
plate_width = xmax - xmin  # 290 - 175 = 115
plate_height = ymax - ymin  # 255 - 228 = 27
center_x = (xmin + xmax) / 2  # (175 + 290) / 2 = 232.5
center_y = (ymin + ymax) / 2  # (228 + 255) / 2 = 241.5
area = plate_width * plate_height  # 115 * 27 = 3105
```

### Data Statistics

```
Total Records: 400+
Image Coverage: 243 unique images
Entries per Image: 1-2 (mostly 1)

Bounding Box Ranges:
  X coordinates: 0 - ~800 pixels
  Y coordinates: 0 - ~600 pixels

Plate Size Ranges:
  Width: ~40-200 pixels
  Height: ~15-60 pixels
  Area: ~600-12000 pixels²
```

### File Characteristics

- **Format:** CSV (Comma-Separated Values)
- **Encoding:** UTF-8
- **Row Count:** 400+ data rows + 1 header row
- **Column Count:** 5
- **File Size:** ~50-100 KB
- **Delimiter:** Comma (,)
- **Quote Character:** None (numeric fields)

### Usage Example

```python
import pandas as pd

# Load CSV
df = pd.read_csv('labels.csv')

# Calculate plate statistics
df['width'] = df['xmax'] - df['xmin']
df['height'] = df['ymax'] - df['ymin']
df['area'] = df['width'] * df['height']

# Statistics
print(f"Average plate width: {df['width'].mean():.2f} pixels")
print(f"Average plate height: {df['height'].mean():.2f} pixels")
print(f"Average plate area: {df['area'].mean():.2f} pixels²")

# Distribution
print(df['width'].describe())
```

---

## 4. Object Detection Model: object_detection.h5

### Purpose
Trained deep learning model for license plate detection in images.

### File Format

**Type:** H5 (HDF5 - Hierarchical Data Format)
**Framework:** Keras/TensorFlow
**Extension:** .h5

### File Structure (Internal)

```
object_detection.h5
├── model_weights/          # Learned parameters
│   ├── conv layers (weights & biases)
│   ├── dense layers (weights & biases)
│   └── ...
├── model_config/           # Architecture definition
│   ├── layer definitions
│   ├── connections
│   └── hyperparameters
├── metadata/               # File information
│   ├── keras_version: 2.11.0
│   ├── backend: tensorflow
│   └── training_config
└── optimizer_state/        # (Optional) Training state
```

### Model Metadata

```
Keras Version: 2.11.0
Backend: TensorFlow
Model Type: Sequential/Functional
Input Shape: (None, 640, 640, 3)
Output Shape: (None, num_detections, 4)
Parameters: ~50-100 million (estimated for YOLO)
```

### How Model Works

**Input:**
- Image (640 x 640 pixels, RGB)

**Processing:**
- Passes through convolutional layers
- Feature extraction
- Detection head analysis

**Output:**
- Bounding boxes [x, y, w, h]
- Confidence scores (0-1)
- Class probabilities

### Loading the Model

```python
import tensorflow as tf

# Load model from H5 file
model = tf.keras.models.load_model('object_detection.h5')

# Get model summary
model.summary()

# Perform inference
predictions = model.predict(image_array)
```

### File Size

- **Typical Size:** 200-500 MB (depends on model complexity)
- **Compression:** Optional (HDF5 can compress)
- **Storage:** Stored on disk, loaded into RAM for use

### Training Information

- **Training Framework:** TensorFlow with Keras API
- **Loss Function:** YOLO loss (localization + confidence + classification)
- **Optimizer:** Adam or SGD
- **Epochs:** ~100 (as per configuration)
- **Batch Size:** 8-16 (configurable)

---

## 5. Event Log Files: object_detection/

### Purpose
TensorFlow training event logs for monitoring model training progress.

### Location & Structure

```
object_detection/
├── train/
│   └── events.out.tfevents.1653911242.*.v2
└── validation/
    └── events.out.tfevents.1653911305.*.v2
```

### File Format

**Type:** Binary TensorFlow event files
**Format:** Protocol Buffer (.pb or .v2)
**Readable With:** TensorBoard

### Contents (Examples)

**Training Events:**
- Loss over time
- Accuracy metrics
- Learning rate changes
- Batch statistics
- Gradient norms

**Validation Events:**
- Validation loss
- Validation accuracy
- mAP (mean Average Precision)
- Precision/Recall curves

### Viewing Events

```bash
# Launch TensorBoard
tensorboard --logdir=object_detection

# Access in browser at http://localhost:6006
```

### Timestamp Information

From filename: `events.out.tfevents.1653911242.*.v2`

```
1653911242 = Unix timestamp
         = May 31, 2022, ~10:47 AM UTC
```

### Statistics Available

```
Scalars:
  - loss/total
  - loss/localization
  - loss/confidence
  - metrics/accuracy
  - learning_rate

Distributions:
  - weights
  - activations
  - gradients
  
Histograms:
  - layer activations
  - weight distributions
```

---

## 6. YOLOv5 Model Artifacts: results/yolov5/

### Components

#### yolov5s.pt
- **Uninitialized YOLOv5 Small model weights
- **Size:** ~200 MB
- **Purpose:** Starting point for transfer learning

#### Data Files
```
data/
├── coco.yaml
├── coco128.yaml
├── VOC.yaml
└── [other dataset configs]
```

#### Run Outputs
```
runs/train/
├── weights/best.pt        # Best model weights
├── weights/last.pt        # Last checkpoint
└── metrics/                # Performance metrics
```

---

## 7. Data Statistics Summary

### Overall Project Data

```
Source Images:        243
Training Samples:     224 (92.2%)
Testing Samples:      25  (10.3%)
Unused:              ~5-10 (2-5%)

PASCAL VOC Files:     243 XML files
YOLO Format Files:    ~250 TXT files
CSV Records:          400+ entries

Total Disk Usage:
  - XML annotations: ~2-3 MB
  - YOLO labels:     ~15 KB
  - CSV data:        ~50-100 KB
  - Models:          200-500 MB
  - Event logs:      10-50 MB
  - Total:           ~200-600 MB
```

---

## 8. Format Comparison

### PASCAL VOC (XML)
**Advantages:**
- Human readable
- Detailed metadata (truncated, difficult flags)
- Widely supported
- Easy to parse

**Disadvantages:**
- Verbose (text-based)
- Slower to parse
- Larger file size

### YOLO Format (TXT)
**Advantages:**
- Efficient binary format
- Fast parsing
- Smaller file size
- Normalized coordinates (scale-invariant)

**Disadvantages:**
- Less metadata
- Requires header information
- Less human readable

### CSV Format
**Advantages:**
- Simple tabular format
- Excel compatible
- Easy analysis with pandas

**Disadvantages:**
- Denormalized (no coordinate scaling)
- Missing metadata
- Redundant if multiple objects per image

---

## Data Workflow

```
Source Images + PASCAL VOC Annotations
    ↓
p01_parse_annotations.py  [Extract XML]
    ↓
p02_preprocess_dataset.py  [Convert to YOLO]
    ↓
p03_generate_labels.py  [Split train/test, copy files]
    ↓
data_images/train/  and  data_images/test/
    ↓
p04_train_model.py  [Train YOLO model]
    ↓
object_detection.h5  [Trained model]
    ↓
p06_inference_pipeline.py  [Run predictions]
    ↓
Results with detections
```

---

## Notes

1. **Path Prefixes:** Most paths shown are relative to project root
2. **File Naming:** Sequential numbering (N1, N2, etc.) maintained across formats
3. **Coordinates:** Always in pixel units, except YOLO format (normalized 0-1)
4. **Single Object:** Most images contain single license plate
5. **Quality:** All annotations manually verified for accuracy
