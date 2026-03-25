# Scripts Documentation - Detailed Analysis

## Overview

This document provides detailed documentation for each Python script in the `anpr_local_pipeline/scripts/` directory.

---

## p01_parse_annotations.py

### Purpose
Parses PASCAL VOC XML annotation files to extract image metadata and bounding box information.

### Location
`anpr_local_pipeline/scripts/p01_parse_annotations.py`

### Main Function

```python
def parsing(annotation_path):
    """
    Parses XML annotation files to extract image metadata and bounding box information.

    Args:
        annotation_path (str): Path to the directory containing XML annotation files.

    Returns:
        pandas.DataFrame: DataFrame with columns: 
            [filename, width, height, name, xmin, ymin, xmax, ymax]
    """
```

### Algorithm

1. **Validate Input**
   - Check if annotation path exists
   - Verify readable/writable permissions

2. **Find XML Files**
   - Scan directory for `.xml` files
   - Build list of files to process

3. **Parse Each File**
   ```xml
   For each XML file:
       ├─ Parse XML tree
       ├─ Extract filename
       ├─ Extract image dimensions (width, height)
       ├─ For each <object>:
       │  ├─ Get object name (class)
       │  └─ Get bounding box (xmin, ymin, xmax, ymax)
       └─ Add to data list
   ```

4. **Aggregate Data**
   - Combine all parsed data
   - Create pandas DataFrame
   - Return to caller

### Key Implementation Details

**XML Parsing:**
```python
import xml.etree.ElementTree as ET

tree = ET.parse(file_path)
root = tree.getroot()
filename = root.find('filename').text
size = root.find('size')
width = int(size.find('width').text)
height = int(size.find('height').text)

for obj in root.findall('object'):
    name = obj.find('name').text
    bndbox = obj.find('bndbox')
    xmin = int(bndbox.find('xmin').text)
    ymin = int(bndbox.find('ymin').text)
    xmax = int(bndbox.find('xmax').text)
    ymax = int(bndbox.find('ymax').text)
```

**Error Handling:**
```python
try:
    # Parse file
except Exception as e:
    print(f"Error parsing file {file_path}: {e}")
    continue  # Continue with next file
```

### Input
- **Type:** Directory path (string)
- **Expected:** Directory containing `.xml` files
- **Format:** PASCAL VOC XML

### Output

**Return Type:** `pandas.DataFrame`

**Example Output:**
```python
     filename  width  height         name  xmin  ymin  xmax  ymax
0      N1.jpg    480     360  license_plate   175   228   290   255
1      N2.jpg    640     480  license_plate   262   141   318   160
2      N3.jpg    800     600  license_plate    87   166   216   212
...
```

### Error Scenarios

| Scenario | Handling |
|----------|----------|
| Path doesn't exist | Print error, return empty DataFrame |
| No XML files found | Print warning, return empty DataFrame |
| Malformed XML | Print error, continue to next file |
| Missing XML elements | Catch exception, skip that file |

### Dependencies
```python
import xml.etree.ElementTree as ET
import os
import pandas as pd
```

### Usage Example
```python
from scripts.p01_parse_annotations import parsing

annotation_path = 'data/annotations/'
df = parsing(annotation_path)
print(f"Parsed {len(df)} annotations")
print(df.head())
```

---

## p02_preprocess_dataset.py

### Purpose
Converts bounding box coordinates from PASCAL VOC format to YOLO format (normalized).

### Location
`anpr_local_pipeline/scripts/p02_preprocess_dataset.py`

### Main Function

```python
def convert_to_yolo_format(df, class_mapping={'license_plate': 0}):
    """
    Converts bounding box coordinates to YOLO format (normalized).

    Args:
        df (pd.DataFrame): DataFrame with columns 
            [filename, width, height, name, xmin, ymin, xmax, ymax]
        class_mapping (dict): Maps class names to integer IDs

    Returns:
        pd.DataFrame: DataFrame with YOLO formatted annotations:
            [class_id, center_x_norm, center_y_norm, width_norm, height_norm]
    """
```

### Conversion Algorithm

**Step 1: Filter by Class**
```python
df_filtered = df[df['name'].isin(class_mapping.keys())].copy()
```

**Step 2: Map Class to ID**
```python
df_filtered['class_id'] = df_filtered['name'].map(class_mapping)
# license_plate → 0
```

**Step 3: Calculate Center and Dimensions**
```python
# Calculate in pixel coordinates
center_x = (xmin + xmax) / 2
center_y = (ymin + ymax) / 2
width = xmax - xmin
height = ymax - ymin
```

**Step 4: Normalize to 0-1 Range**
```python
center_x_norm = center_x / image_width
center_y_norm = center_y / image_height
width_norm = width / image_width
height_norm = height / image_height
```

### Conversion Example

**Input (PASCAL VOC):**
```
Image: N1.jpg
Width: 480, Height: 360
xmin: 175, ymin: 228, xmax: 290, ymax: 255
```

**Processing:**
```python
center_x = (175 + 290) / 2 = 232.5
center_y = (228 + 255) / 2 = 241.5
width = 290 - 175 = 115
height = 255 - 228 = 27

center_x_norm = 232.5 / 480 = 0.4844
center_y_norm = 241.5 / 360 = 0.6708
width_norm = 115 / 480 = 0.2396
height_norm = 27 / 360 = 0.0750
```

**Output (YOLO):**
```
class_id: 0
center_x_norm: 0.4844
center_y_norm: 0.6708
width_norm: 0.2396
height_norm: 0.0750
```

### YOLO Format Advantages

1. **Scale Invariant:** Works with images of any size
2. **Normalized:** Values between 0 and 1
3. **Center-based:** Uses center point instead of corners
4. **Efficient:** Compatible with YOLO training pipeline

### Input

**Type:** `pandas.DataFrame`

**Expected Columns:**
```python
['filename', 'width', 'height', 'name', 'xmin', 'ymin', 'xmax', 'ymax']
```

### Output

**Type:** `pandas.DataFrame`

**New Columns Added:**
```python
['class_id', 'center_x_norm', 'center_y_norm', 'width_norm', 'height_norm']
```

### Error Handling

```python
if df.empty:
    return pd.DataFrame(columns=[...])  # Empty DataFrame

if 'filename' not in df.columns:
    logging.error("'filename' column missing")
    return

if df_filtered.empty:
    print("Warning: No relevant classes found")
    return
```

### Class Mapping Customization

**Default:**
```python
class_mapping = {'license_plate': 0}
```

**Extend for Multiple Classes:**
```python
class_mapping = {
    'license_plate': 0,
    'vehicle_body': 1,
    'vehicle_wheel': 2
}

df_yolo = convert_to_yolo_format(df, class_mapping)
```

### Dependencies
```python
import pandas as pd
```

### Usage Example
```python
from scripts.p02_preprocess_dataset import convert_to_yolo_format
from scripts.p01_parse_annotations import parsing

# Parse annotations
df = parsing('data/annotations/')

# Convert to YOLO format
df_yolo = convert_to_yolo_format(df)
print(f"Converted {len(df_yolo)} annotations")
print(df_yolo.head())
```

---

## p03_generate_labels.py

### Purpose
Splits dataset into training/testing sets, generates YOLO label files, and copies images.

### Location
`anpr_local_pipeline/scripts/p03_generate_labels.py`

### Main Function

```python
def generate_yolo_labels(df_yolo, output_image_dir_base, 
                         output_label_dir_base, source_image_dir,
                         test_size=0.2, random_state=42):
    """
    Splits data into train/test sets, copies images, and saves YOLO label files.

    Args:
        df_yolo: DataFrame with YOLO format annotations
        output_image_dir_base: Base directory for train/test images
        output_label_dir_base: Base directory for train/test labels
        source_image_dir: Source directory containing images
        test_size: Proportion of test set (0.2 = 20%)
        random_state: Seed for reproducibility
    """
```

### Algorithm

**Step 1: Extract Unique Images**
```python
unique_files = df_yolo['filename'].unique()
```

**Step 2: Train/Test Split**
```python
from sklearn.model_selection import train_test_split

train_files, test_files = train_test_split(
    unique_files,
    test_size=test_size,
    random_state=random_state
)
```

**Step 3: Create Directory Structure**
```
data_images/
├── train/  [copies of training images]
└── test/   [copies of test images]
```

**Step 4: Filter Data by Split**
```python
train_df = df_yolo[df_yolo['filename'].isin(train_files)]
test_df = df_yolo[df_yolo['filename'].isin(test_files)]
```

**Step 5: Save YOLO Label Files**
```
For each train file:
    Create: data_images/train/FILENAME.txt
    Content: "<class_id> <center_x_norm> <center_y_norm> <width_norm> <height_norm>"

For each test file:
    Create: data_images/test/FILENAME.txt
    Content: "<class_id> <center_x_norm> <center_y_norm> <width_norm> <height_norm>"
```

**Step 6: Copy Images**
```python
shutil.copy(source_path, dest_path)
```

### Train/Test Split Configuration

**Default:**
```python
test_size = 0.2      # 20% test, 80% train
random_state = 42    # Fixed seed for reproducibility
```

**Impact on 243 Images:**
```
Train: 243 * 0.8 = 194.4 ≈ 194 images (80%)
Test:  243 * 0.2 = 48.6 ≈ 49 images (20%)
```

### YOLO Label File Format

**File Name:** `{image_name}.txt`

**Content (one line per object):**
```
class_id center_x_norm center_y_norm width_norm height_norm
0 0.4844 0.6708 0.2396 0.0750
```

**File Size:** ~30-50 bytes per image (single plate)

### Output Directory Structure

After execution:
```
data_images/
├── train/
│   ├── N100.txt       ← Label file
│   ├── N101.txt
│   ├── N100.jpg       ← Image file
│   ├── N101.jpg
│   └── ... (194 files each)
│
└── test/
    ├── N1.txt         ← Label file
    ├── N2.txt
    ├── N1.jpg         ← Image file
    ├── N2.jpg
    └── ... (49 files each)
```

### Logging

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.info(f"Splitting {len(unique_files)} images...")
logging.info(f"Train: {len(train_files)} images")
logging.info(f"Test: {len(test_files)} images")
```

### Error Handling

```python
if df_yolo.empty:
    logging.error("Input DataFrame is empty")
    return

if 'filename' not in df_yolo.columns:
    logging.error("'filename' column missing")
    return

if len(unique_files) == 1:
    logging.warning(f"Only {len(unique_files)} image found")
    train_files = unique_files
    test_files = []
```

### Edge Cases

| Condition | Handling |
|-----------|----------|
| Empty DataFrame | Return with error |
| Single image | Put in train (no test) |
| Small datasets | Use available samples |
| Existing files | Overwrite or skip |
| Read-only dirs | Log warning, continue |

### Dependencies
```python
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import shutil
import logging
```

### Usage Example
```python
from scripts.p03_generate_labels import generate_yolo_labels
from scripts.p01_parse_annotations import parsing
from scripts.p02_preprocess_dataset import convert_to_yolo_format

# Parse and preprocess
df = parsing('data/annotations/')
df_yolo = convert_to_yolo_format(df)

# Generate labels and split
generate_yolo_labels(
    df_yolo,
    output_image_dir_base='data_images',
    output_label_dir_base='data_images',
    source_image_dir='data/images',
    test_size=0.2
)
```

---

## p04_train_model.py

### Purpose
Train YOLOv5 model on the prepared dataset.

### Location
`anpr_local_pipeline/scripts/p04_train_model.py`

### Key Functions

**Clone Repository:**
```python
def clone_yolov5_repo():
    """Clones YOLOv5 repository if not already present."""
```

**Install Requirements:**
```python
def install_yolov5_requirements():
    """Installs requirements for YOLOv5."""
```

**Train Model:**
```python
def train_model():
    """Executes YOLO training process."""
```

### Paths and Configuration

```python
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
YOLOV5_DIR = os.path.join(PROJECT_ROOT, 'yolov5')
DATA_YAML_PATH = os.path.join(PROJECT_ROOT, 'data.yaml')

# Training parameters
DEFAULT_YOLO_MODEL_CFG = 'yolov5s.yaml'  # Small model
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 100
DEFAULT_RUN_NAME = 'license_plate_model'
DEFAULT_IMG_SIZE = 640
```

### Clone Process

```python
def clone_yolov5_repo():
    if not os.path.exists(YOLOV5_DIR):
        subprocess.run([
            'git', 'clone',
            'https://github.com/ultralytics/yolov5.git',
            YOLOV5_DIR
        ], check=True, capture_output=True, text=True)
        logging.info("YOLOv5 repository cloned successfully.")
    else:
        logging.info(f"YOLOv5 already exists at {YOLOV5_DIR}")
```

### Requirements Installation

```python
def install_yolov5_requirements():
    requirements_path = os.path.join(YOLOV5_DIR, 'requirements.txt')
    if not os.path.exists(requirements_path):
        logging.error("requirements.txt not found")
        return False
    
    subprocess.run(
        ['pip', 'install', '-r', requirements_path],
        check=True,
        cwd=YOLOV5_DIR
    )
```

### Training Execution

**Training Command:**
```bash
python yolov5/train.py \
    --data data.yaml \
    --weights yolov5s.pt \
    --epochs 100 \
    --batch-size 8 \
    --imgsz 640 \
    --name license_plate_model
```

**Parameter Descriptions:**

| Parameter | Value | Meaning |
|-----------|-------|---------|
| --data | data.yaml | Dataset configuration |
| --weights | yolov5s.pt | Starting weights (pre-trained) |
| --epochs | 100 | Training iterations |
| --batch-size | 8 | Samples per batch |
| --imgsz | 640 | Input image size |
| --name | license_plate_model | Run name for outputs |

### Output Files

**Training Generates:**
```
yolov5/runs/train/license_plate_model/
├── weights/
│   ├── best.pt      ← Best model (lowest val loss)
│   └── last.pt      ← Final checkpoint
├── results.csv      ← Training metrics
├── confusion_matrix.png
├── F1_curve.png
├── PR_curve.png
├── P_curve.png
├── R_curve.png
├── results.png      ← Training curves
└── events          ← TensorBoard events
```

### Training Metrics

**Tracked Metrics:**
```
- Loss (total, localization, objectness, classification)
- Precision: True Positives / (True Pos + False Pos)
- Recall: True Positives / (True Pos + False Neg)
- mAP@.5: Mean Average Precision at IoU=0.5
- mAP@.5:.95: Mean Average Precision at IoU=0.5 to 0.95
```

### Training Behavior

**Learning Process:**
1. Load ImageNet pre-trained weights
2. Modify final layer for 1 class (license_plate)
3. Initialize optimizer (SGD + momentum)
4. For each epoch:
   - Forward pass through all training batches
   - Calculate loss
   - Backward pass (gradients)
   - Update weights
   - Validate on test set
   - Log metrics

**Early Stopping:**
- No explicit early stopping in basic config
- Can be added: Stop if val_loss doesn't improve for N epochs

### Typical Training Duration

```
Total Epochs: 100
Images per Epoch: 224
Batch Size: 8
Batches per Epoch: 224 / 8 = 28

Time per Epoch: ~2-5 minutes (depends on hardware)
Total Time: 100 * 3 min = ~300 minutes = 5 hours
```

### Current Status

**Note:** Due to environment constraints, this script may:
- Skip actual training
- Use placeholder models
- Load existing weights instead
- Output simulation results

### Error Handling

```python
try:
    clone_yolov5_repo()
    install_yolov5_requirements()
    train_model()
    logging.info("Training completed successfully")
except subprocess.CalledProcessError as e:
    logging.error(f"Process failed: {e}")
    raise
except FileNotFoundError as e:
    logging.error(f"File not found: {e}")
    raise
```

### Dependencies
```python
import subprocess       # Run external commands
import logging
import os
import shutil
```

### Usage Example
```python
from scripts.p04_train_model import train_model

try:
    train_model()
except Exception as e:
    print(f"Training failed: {e}")
```

---

## p05_export_model.py

### Purpose
Export trained YOLOv5 model to multiple formats (ONNX, TorchScript) for deployment.

### Key Functions

```python
def export_model(model_path, output_dir):
    """
    Exports trained model to ONNX and TorchScript formats.
    """

def export_to_onnx(model_path, output_path):
    """
    Exports to ONNX format.
    """

def export_to_torchscript(model_path, output_path):
    """
    Exports to TorchScript format.
    """
```

### Export Formats

**1. ONNX (Open Neural Network Exchange)**

- **Purpose:** Framework-agnostic model format
- **File Extension:** `.onnx`
- **Size:** ~200-300 MB
- **Runtime:** Multiple (ONNX Runtime, TritOn, TensorRT)
- **Use Case:** Cloud deployment, cross-platform inference

**2. TorchScript**

- **Purpose:** PyTorch serialization format
- **File Extension:** `.pt` or `.torchscript`
- **Size:** ~200-300 MB
- **Runtime:** PyTorch runtime
- **Use Case:** PyTorch ecosystem applications

### ONNX Export Process

```python
import torch
import onnx

# Load model
model = torch.load('best.pt')
model.eval()

# Create dummy input (same shape as training input)
dummy_input = torch.randn(1, 3, 640, 640)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    'model.onnx',
    opset_version=12,
    input_names=['images'],
    output_names=['output'],
    dynamic_axes={
        'images': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
```

### TorchScript Export Process

```python
import torch

# Load and trace model
model = torch.load('best.pt')
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 640, 640)

# Trace the model
scripted_model = torch.jit.trace(model, dummy_input)

# Save to file
torch.jit.save(scripted_model, 'model.torchscript')
```

### Output Artifacts

**ONNX Model:**
- `model.onnx` - Trained model in ONNX format
- ~200-300 MB
- Can be optimized further

**TorchScript Model:**
- `model.torchscript` or `model.pt`
- ~200-300 MB
- Ready for deployment

### ONNX Validation

```python
import onnx

# Load and check ONNX model
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
print("ONNX model is valid")
```

### Deployment Considerations

**ONNX Advantages:**
- ✅ Supports multiple runtimes
- ✅ Optimized for production
- ✅ Cross-platform
- ✅ Hardware acceleration support

**TorchScript Advantages:**
- ✅ Pure PyTorch
- ✅ No external dependencies
- ✅ Dynamic control flow support
- ✅ Easy to debug

### Model Compatibility

**Input Specification:**
```python
Input Shape: (Batch Size, 3 Channels, 640 Height, 640 Width)
Data Type: float32
Normalized: [-1, 1] or [0, 1] (depends on preprocessing)
```

**Output Specification:**
```
Detection tensor with shape: (Batch Size, Num Detections, Detection Info)
Detection Info: [x, y, w, h, confidence, class_scores...]
```

### Current Status

Like training, this script may:
- Use placeholder models
- Skip actual export
- Create dummy artifacts
- Output simulation results

### Error Handling

```python
try:
    export_to_onnx(model_path, onnx_output)
    export_to_torchscript(model_path, ts_output)
except Exception as e:
    logging.error(f"Export failed: {e}")
    raise
```

### Dependencies
```python
import torch                    # PyTorch
import onnx                     # ONNX utilities
import logging
import os
```

---

## p06_inference_pipeline.py

### Purpose
Load exported model and run inference to detect license plates in images.

### Key Functions

```python
def load_model(model_path):
    """Loads exported ONNX or TorchScript model."""

def preprocess_image(image_path, target_size=(640, 640)):
    """Preprocesses image for inference."""

def run_inference(image, model, conf_threshold=0.5):
    """Runs model inference on image."""

def nms(detections, iou_threshold=0.4):
    """Apply Non-Maximum Suppression."""

def postprocess_detections(raw_detections, image_shape):
    """Converts output to bounding box format."""
```

### Inference Pipeline

**Step 1: Load Model**
```python
import onnxruntime as rt

session = rt.InferenceSession('model.onnx')
```

**Step 2: Load Image**
```python
import cv2
image = cv2.imread('test_image.jpg')
```

**Step 3: Preprocess**
```python
# Resize to 640x640
image_resized = cv2.resize(image, (640, 640))

# Normalize to [0, 1]
image_normalized = image_resized / 255.0

# Convert to NCHW format
image_tensor = image_normalized.transpose(2, 0, 1)
image_tensor = image_tensor[np.newaxis, :]
```

**Step 4: Run Inference**
```python
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

raw_output = session.run([output_name], {input_name: image_tensor})
```

**Step 5: Post-Processing**
```python
# Apply confidence threshold
detections = raw_output[0]
confident_detections = [d for d in detections if d[4] > 0.5]

# Apply NMS (remove duplicates)
final_detections = nms(confident_detections)

# Scale back to original image size
scaled_detections = scale_to_original(final_detections, original_shape)
```

### Non-Maximum Suppression (NMS)

**Purpose:** Remove overlapping/duplicate detections

**Algorithm:**
1. Sort detections by confidence (highest first)
2. For each detection:
   - Keep if no overlap with higher-confidence boxes
   - Compute IoU with all previously kept detections
   - Remove if IoU > threshold

**IoU Calculation:**
```python
def iou(box1, box2):
    """
    Intersection over Union calculation.
    box = [x1, y1, x2, y2]
    """
    intersection = calculate_intersection(box1, box2)
    union = calculate_union(box1, box2)
    return intersection / union
```

### Detection Output Format

```python
[
    {
        'confidence': 0.95,
        'class_id': 0,
        'class_name': 'license_plate',
        'bbox': [x1, y1, x2, y2],  # pixel coordinates
        'center': (cx, cy),
        'width': w,
        'height': h
    },
    ...
]
```

### Performance Metrics

**Inference Speed:**
```
- Single image: ~50-100 ms
- Batch of 10: ~150-300 ms

Depends on:
- Hardware (GPU vs CPU)
- Model size (yolov5s vs yolov5l)
- Image resolution
- Number of detected objects
```

### Visualization

```python
def draw_detections(image, detections):
    """Draw boxes and labels on image."""
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        label = f"{det['class_name']}: {conf:.2f}"
        cv2.putText(image, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image
```

### Batch Inference

```python
def batch_inference(image_dir, model, batch_size=10):
    """Process multiple images efficiently."""
    
    results = {}
    batch = []
    
    for img_file in os.listdir(image_dir):
        # Load and preprocess
        image = load_image(img_file)
        batch.append(image)
        
        if len(batch) == batch_size:
            # Run inference on batch
            detections = session.run([output_name],
                                    {input_name: batch})
            # Process results
            results.update(process_batch_output(detections))
            batch = []
    
    # Handle remaining images
    if batch:
        detections = session.run([output_name],
                                {input_name: batch})
        results.update(process_batch_output(detections))
    
    return results
```

### Current Status

This script may:
- Use placeholder models
- Return simulated detections
- Skip actual inference
- Provide dummy results for testing

### Dependencies
```python
import onnxruntime        # ONNX inference
import cv2               # Image processing
import numpy as np       # Numerical operations
import logging
```

### Usage Example
```python
from scripts.p06_inference_pipeline import run_inference

detections = run_inference('test_image.jpg', model)
for det in detections:
    print(f"Found {det['class_name']} at confidence {det['confidence']:.2f}")
```

---

## 07_ocr_utils.py

### Purpose
Extract text from license plate images using Tesseract OCR.

### Key Functions

```python
def preprocess_for_ocr(image_np):
    """
    Preprocesses image crop for better OCR results.
    Applies contrast enhancement and filtering.
    """

def extract_text_from_plate(image_np, config=''):
    """
    Uses Tesseract to extract text from plate image.
    Returns extracted text string.
    """
```

### Image Preprocessing

**Step 1: Color Conversion**
```python
# Convert BGR to RGB (OpenCV uses BGR)
image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

# Convert to PIL Image
pil_image = Image.fromarray(image_rgb)
```

**Step 2: Grayscale Conversion**
```python
gray_image = pil_image.convert('L')
```

**Step 3: Contrast Enhancement**
```python
from PIL import ImageEnhance

enhancer = ImageEnhance.Contrast(gray_image)
enhanced_image = enhancer.enhance(2.0)  # Increase contrast 2x
```

**Step 4: Optional Filtering**
```python
# Median filter (denoising)
from PIL import ImageFilter
filtered = enhanced_image.filter(ImageFilter.MedianFilter(size=3))

# Binarization (thresholding)
# binary = enhanced.point(lambda x: 0 if x < 128 else 255, '1')
```

### Full Preprocessing Function

```python
def preprocess_for_ocr(image_np):
    """
    Args:
        image_np (numpy.ndarray): Image in BGR format from OpenCV
    
    Returns:
        PIL.Image: Preprocessed image for OCR
    """
    try:
        # BGR → RGB
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Grayscale
        gray = pil_image.convert('L')
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(2.0)
        
        # Optional: Denoising
        # final = enhanced.filter(ImageFilter.MedianFilter(3))
        
        return enhanced
    
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        # Fallback: minimal preprocessing
        return pil_image.convert('L')
```

### OCR Extraction

**Basic Extraction:**
```python
import pytesseract

text = pytesseract.image_to_string(pil_image)
print(f"Extracted: {text}")  # "ABC123DE"
```

**With Tesseract Configuration:**
```python
text = pytesseract.image_to_string(
    pil_image,
    config='--psm 8'  # Single character mode
)

# PSM (Page Segmentation Mode) options:
# 0 = Orientation and script detection only
# 3 = Fully automatic page segmentation (DEFAULT)
# 4 = Assume single column  
# 6 = Assume uniform block
# 7 = Treat as single text line
# 8 = Treat as single word
# 11 = Sparse text
# 13 = Raw line
```

**Confidence Scores:**
```python
result = pytesseract.image_to_data(pil_image, output_type='dict')

text_list = result['text']
confidence_list = result['conf']

for text, conf in zip(text_list, confidence_list):
    if text.strip():  # Skip empty
        print(f"{text}: {conf}%")
```

### Full Extraction Function

```python
def extract_text_from_plate(image_crop):
    """
    Args:
        image_crop (numpy.ndarray): Cropped plate image
    
    Returns:
        str: Extracted license plate text
    """
    try:
        # Preprocess
        preprocessed = preprocess_for_ocr(image_crop)
        
        # Extract text
        text = pytesseract.image_to_string(
            preprocessed,
            config='--psm 8'
        )
        
        # Clean output
        text = text.strip()
        
        logging.info(f"OCR Result: {text}")
        return text
    
    except Exception as e:
        logging.error(f"OCR extraction failed: {e}")
        return ""
```

### Integration with Detection

```python
def extract_plates_text(image_path, detections):
    """
    Extract text from all detected plates.
    
    Args:
        image_path: Path to original image
        detections: List of detection dictionaries
    
    Returns:
        List of dicts with plate text
    """
    image = cv2.imread(image_path)
    results = []
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        
        # Crop plate region
        plate_crop = image[y1:y2, x1:x2]
        
        # Extract text
        text = extract_text_from_plate(plate_crop)
        
        results.append({
            'bbox': det['bbox'],
            'confidence': det['confidence'],
            'plate_text': text
        })
    
    return results
```

### Tesseract Installation

**macOS:**
```bash
brew install tesseract
```

**Ubuntu/Debian:**
```bash
apt-get install tesseract-ocr
```

**Windows:**
- Download installer from: https://github.com/UB-Mannheim/tesseract/wiki

**Verify Installation:**
```bash
tesseract --version
```

### OCR Quality Enhancement Tips

1. **Preprocessing:**
   - Increase contrast
   - Remove noise
   - Ensure good lighting

2. **Image Size:**
   - Minimum: 30x30 pixels
   - Optimal: 300x300 pixels
   - Larger = better accuracy (to a point)

3. **Orientation:**
   - Ensure plate is upright
   - Tesseract can handle slight rotation

4. **Language:**
   - Train custom models for specific plates
   - Use multiple language packs for international plates

### Typical Accuracy

**Ideal Conditions:**
- Clean, well-lit plate images
- Upright orientation
- No partial occlusion
- **Accuracy: 95%+**

**Real-World Conditions:**
- Variable lighting
- Different angles
- Partial occlusion
- **Accuracy: 70-85%**

### Current Status

OCR utility is available but requires:
- Tesseract engine installed on system
- Pytesseract Python package
- May not work in all environments

### Error Handling

```python
try:
    pytesseract.pytesseract.pytesseract_cmd = 'tesseract'
except:
    logging.warning("Tesseract not found in PATH")
```

### Dependencies
```python
import pytesseract              # OCR interface
from PIL import Image           # Image processing
from PIL import ImageEnhance    # Image enhancement
from PIL import ImageFilter     # Filtering
import cv2                      # OpenCV
import numpy as np              # Numerical
import logging                  # Logging
```

### Usage Example
```python
from scripts.p07_ocr_utils import extract_text_from_plate
import cv2

# Load plate image
plate_image = cv2.imread('license_plate_crop.jpg')

# Extract text
text = extract_text_from_plate(plate_image)
print(f"License Plate: {text}")
```

---

## Summary Table

| Script | Input | Output | Purpose |
|--------|-------|--------|---------|
| p01 | XML files | DataFrame | Parse annotations |
| p02 | DataFrame | YOLO DataFrame | Convert format |
| p03 | YOLO DF | Text files | Split & save |
| p04 | Train files | best.pt | Train model |
| p05 | best.pt | .onnx, .pt | Export model |
| p06 | Image | Detections | Run inference |
| 07 | Plate crop | Text string | Extract text |

---

## Execution Flow

```
p01 → p02 → p03 → p04 → p05 → p06 → 07
```

Each script depends on the previous one's output.

---

## Notes

- All scripts include comprehensive logging
- Error handling for robustness  
- Modular design for easy testing
- Compatible with future Azure ML migration
