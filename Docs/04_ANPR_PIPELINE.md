# ANPR Local Pipeline Architecture Documentation

## Overview

The ANPR (Automatic Number Plate Recognition) Local Pipeline is a modular, production-ready implementation of a machine learning pipeline. It follows separation of concerns principle with each stage as independent Python script.

---

## Pipeline Architecture

### High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Raw Data                          │
│  Images: images/  |  Annotations: annotations/              │
└─────────────┬───────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  p01_parse_annotations.py - Extract & Parse XML             │
│  - Reads PASCAL VOC XML files                               │
│  - Extracts image metadata and bounding boxes               │
│  - Outputs: pandas DataFrame                                │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  p02_preprocess_dataset.py - Normalize Coordinates          │
│  - Converts PASCAL VOC to YOLO format                       │
│  - Normalizes coordinates (0-1 range)                       │
│  - Outputs: YOLO-formatted DataFrame                        │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  p03_generate_labels.py - Split & Generate Labels           │
│  - Splits data into train/test                              │
│  - Creates directory structure                              │
│  - Saves YOLO label files (.txt)                            │
│  - Copies images to train/test folders                      │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTPUT: Prepared Data                                       │
│  data_images/train/ ──── (224 images + labels)              │
│  data_images/test/  ──── (25 images + labels)               │
│  data.yaml ────────────── (configuration file)              │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  p04_train_model.py - Train YOLOv5 Model                    │
│  - Clone YOLOv5 repository                                  │
│  - Install dependencies                                     │
│  - Train model on prepared data                             │
│  - Save best model weights                                  │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTPUT: Trained Model                                       │
│  best.pt (trained weights)                                   │
│  last.pt (final checkpoint)                                  │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  p05_export_model.py - Export for Deployment                │
│  - Load trained model                                       │
│  - Export to ONNX format                                     │
│  - Export to TorchScript format                              │
│  - Create inference-ready artifacts                         │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  OUTPUT: Exported Models                                     │
│  model.onnx (ONNX format)                                    │
│  model.torchscript (TorchScript format)                      │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  p06_inference_pipeline.py - Run Inference                  │
│  - Load exported model                                      │
│  - Process input image                                      │
│  - Detect license plates                                    │
│  - Apply NMS (Non-Max Suppression)                           │
│  - Return detections with confidence                        │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│  07_ocr_utils.py - Extract Plate Text                       │
│  - Crop detected plates from image                          │
│  - Preprocess for OCR                                       │
│  - Use Tesseract for text extraction                        │
│  - Return plate text                                        │
└─────────────┬──────────────────────────────────────────────┘
              ▼
┌──────────────────────────────────────────────────────────────┐
│                      OUTPUT: Results                         │
│         Detected plates with recognized text                │
└──────────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages (Detailed)

### Stage 1: p01_parse_annotations.py

**Purpose:** Extract data from XML annotation files

**Input:**
- Directory path: `data/annotations/`
- XML files in PASCAL VOC format

**Processing:**
```python
def parsing(annotation_path):
    """
    Parses XML files and extracts annotations.
    Returns DataFrame with columns:
    [filename, width, height, name, xmin, ymin, xmax, ymax]
    """
```

**Algorithm:**
1. Scan annotation directory for XML files
2. For each XML file:
   - Parse XML structure
   - Extract image filename and dimensions
   - Extract all bounding box objects
   - Record coordinates (xmin, ymin, xmax, ymax)
3. Combine all data into DataFrame

**Output DataFrame:**
```
   filename  width  height         name  xmin  ymin  xmax  ymax
0    N1.jpg    480     360  license_plate   175   228   290   255
1    N2.jpg    640     480  license_plate   262   141   318   160
...
```

**Error Handling:**
- Check if annotation path exists
- Verify XML files are present
- Catch parsing exceptions per file
- Continue with remaining files on error

---

### Stage 2: p02_preprocess_dataset.py

**Purpose:** Convert PASCAL VOC coordinates to YOLO format

**Input:**
- DataFrame from Stage 1
- Class mapping: {'license_plate': 0}

**Processing:**
```python
def convert_to_yolo_format(df, class_mapping):
    """
    Converts bounding boxes from PASCAL VOC to YOLO format.
    Output columns: [class_id, center_x_norm, center_y_norm, 
                     width_norm, height_norm]
    """
```

**Conversion Formula:**
```
center_x = (xmin + xmax) / 2
center_y = (ymin + ymax) / 2
width = xmax - xmin
height = ymax - ymin

center_x_norm = center_x / image_width
center_y_norm = center_y / image_height
width_norm = width / image_width
height_norm = height / image_height
```

**Example:**
```
Input (PASCAL VOC): xmin=175, ymin=228, xmax=290, ymax=255, img_w=480, img_h=360
center_x = (175 + 290) / 2 = 232.5
center_y = (228 + 255) / 2 = 241.5
width = 290 - 175 = 115
height = 255 - 228 = 27

center_x_norm = 232.5 / 480 = 0.4844
center_y_norm = 241.5 / 360 = 0.6708
width_norm = 115 / 480 = 0.2396
height_norm = 27 / 360 = 0.0750

Output (YOLO): 0 0.4844 0.6708 0.2396 0.0750
```

**Output DataFrame:**
- Additional columns: class_id, center_x_norm, center_y_norm, width_norm, height_norm
- Filtered to only include 'license_plate' class
- Normalized coordinates in range [0, 1]

---

### Stage 3: p03_generate_labels.py

**Purpose:** Split data and generate training/testing sets

**Input:**
- YOLO-formatted DataFrame from Stage 2
- Parameters: test_size=0.2, random_state=42

**Processing:**
```python
def generate_yolo_labels(df_yolo, output_image_dir_base, 
                        output_label_dir_base, test_size=0.2):
    """
    Splits data into train/test sets.
    Creates directory structure and saves YOLO label files.
    Copies images to appropriate directories.
    """
```

**Steps:**
1. **Get unique images:** Extract list of unique filenames
2. **Split data:** Use sklearn's train_test_split
   - Train: 80% (224 images)
   - Test: 20% (25 images)
3. **Create directories:**
   - `data_images/train/`
   - `data_images/test/`
4. **Copy images:** From source to train/test folders
5. **Save labels:** Create .txt files with YOLO format

**Directory Structure Created:**
```
data_images/
├── train/
│   ├── N100.txt (label)
│   ├── N101.txt (label)
│   ...
│   ├── N100.jpg (image)
│   ├── N101.jpg (image)
│   ...
│
└── test/
    ├── N1.txt (label)
    ├── N2.txt (label)
    ...
    ├── N1.jpg (image)
    ├── N2.jpg (image)
    ...
```

**Label File Content:**
```
# N100.txt
0 0.4844 0.6708 0.2396 0.0750

# N101.txt
0 0.5132 0.6125 0.2458 0.0825
```

---

### Stage 4: p04_train_model.py

**Purpose:** Train YOLO model on prepared dataset

**Input:**
- Prepared data: `data_images/train/`, `data_images/test/`
- Configuration: `data.yaml`
- Model config: `yolov5s.yaml` (small model)

**Processing:**
```python
def clone_yolov5_repo():
    # Clone YOLOv5 repository if not present

def install_yolov5_requirements():
    # Install YOLOv5 dependencies

def train_model():
    # Execute training command via subprocess
```

**Training Parameters:**
```python
DEFAULT_YOLO_MODEL_CFG = 'yolov5s.yaml'
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 100
DEFAULT_RUN_NAME = 'license_plate_model'
DEFAULT_IMG_SIZE = 640
```

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

**Output:**
- `best.pt` - Best model weights (lowest validation loss)
- `last.pt` - Final epoch weights
- Training logs in `runs/train/`

**Training Metrics Generated:**
- Loss (total, box, obj, cls)
- Precision, Recall, mAP@.5, mAP@.5:.95
- Confusion matrix
- Training curve plots

---

### Stage 5: p05_export_model.py

**Purpose:** Export trained model for deployment

**Input:**
- Trained model: `path/to/best.pt`

**Processing:**
```python
def export_model():
    # Load trained YOLO model
    # Export to multiple formats
    # Create inference-ready artifacts
```

**Export Formats:**

1. **ONNX Format**
   - Framework-agnostic
   - Supports multiple inference engines
   - File: `model.onnx`
   - Size: ~200-300 MB

2. **TorchScript Format**
   - PyTorch serialization
   - Can run without Python interpreter
   - File: `model.torchscript` or `model.pt`
   - Size: ~200-300 MB

**Export Process:**
```python
import onnx
import torch

model = torch.load('best.pt')
torch.onnx.export(model, dummy_input, 'model.onnx', ...)
optimized_model = torch.jit.script(model)
torch.jit.save(optimized_model, 'model.torchscript')
```

---

### Stage 6: p06_inference_pipeline.py

**Purpose:** Run inference on images to detect license plates

**Input:**
- Exported model (ONNX or TorchScript)
- Image file or directory

**Processing:**
```python
def inference_pipeline(image_path, model, conf_threshold=0.5):
    """
    Loads image, runs inference, applies NMS.
    Returns detections with confidence scores.
    """
```

**Inference Steps:**
1. **Load Image:** Read from file or stream
2. **Preprocess:** Resize to 640x640, normalize
3. **Inference:** Pass through model
4. **Post-processing:**
   - NMS (Non-Maximum Suppression): Remove duplicate detections
   - Confidence filtering: Keep (score > threshold)
5. **Denormalize:** Convert coordinates back to pixel space

**Output Detections:**
```
[
    {
        'confidence': 0.95,
        'box': [x1, y1, x2, y2],  # pixel coordinates
        'class': 'license_plate'
    },
    ...
]
```

**NMS (Non-Maximum Suppression):**
- **Purpose:** Remove overlapping/duplicate detections
- **Algorithm:** 
  - Sort by confidence descending
  - For each detection:
    - Keep if no overlap with higher-confidence boxes
    - Remove if IoU (Intersection over Union) > threshold

---

### Stage 7: 07_ocr_utils.py

**Purpose:** Extract text from detected license plates

**Input:**
- Image with detected bounding boxes
- Bounding box coordinates

**Processing:**
```python
def preprocess_for_ocr(image_np):
    """
    Preprocesses image crop for better OCR results.
    Applies contrast enhancement and filtering.
    """

def extract_text_from_plate(image_crop):
    """
    Uses Tesseract to extract text from plate image.
    """
```

**OCR Preprocessing Steps:**
1. **Color Conversion:** BGR (OpenCV) → RGB
2. **Grayscale Conversion:** RGB → Grayscale
3. **Contrast Enhancement:** Enhance by factor of 2.0
4. **Denoising:** (Optional) Median filter
5. **Binarization:** (Optional) Thresholding

**Text Extraction:**
```python
import pytesseract

plate_text = pytesseract.image_to_string(preprocessed_image)
```

**OCR Configurations:**
```python
pytesseract.image_to_string(
    image,
    config='--psm 8'  # Single character mode
)
```

**Output:**
- Extracted text string (e.g., "ABC123DE")
- Confidence scores (if available)
- Character bounding boxes (optional)

---

## Configuration: data.yaml

```yaml
train: data_images/train     # Training data path
val: data_images/test        # Validation data path
nc: 1                        # Number of classes
names: ['license_plate']      # Class names
```

**Purpose:**
- Centralized dataset configuration
- Used by YOLO training script
- Specifies relative paths

---

## Main Orchestrator: main.py

**Purpose:** Run all pipeline stages in sequence

**Structure:**
```python
import scripts.p01_parse_annotations as p01
import scripts.p02_preprocess_dataset as p02
import scripts.p03_generate_labels as p03
import scripts.p04_train_model as p04
import scripts.p05_export_model as p05
import scripts.p06_inference_pipeline as p06
import scripts.p07_ocr_utils as p07

# Execute pipeline stages
df = p01.parsing('data/annotations/')
df_yolo = p02.convert_to_yolo_format(df)
p03.generate_yolo_labels(df_yolo, ...)
p04.train_model(...)
p05.export_model(...)
results = p06.inference_pipeline(...)
```

---

## Error Handling & Logging

### Logging Setup

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s'
)
```

### Log Levels

- **INFO:** Normal operation messages
- **WARNING:** Non-fatal issues (missing data, suboptimal config)
- **ERROR:** Serious problems (missing files, import errors)
- **DEBUG:** Detailed diagnostic information

### Common Issues Handled

1. **Missing Files**
   - Annotation files not found
   - Image files missing
   - Model file corrupted

2. **Invalid Data**
   - Malformed XML
   - Incorrect coordinate ranges
   - Empty datasets

3. **Environment Issues**
   - Missing package (YOLOv5)
   - TensorFlow/PyTorch conflicts
   - CUDA/GPU unavailability

---

## Extensibility & Modularity

### Adding New Stages

To add a new pipeline stage:

1. **Create new Python file:** `p0X_new_stage.py`
2. **Define main function:**
   ```python
   def process_data(input_data):
       """Docstring with input/output spec"""
       # Implementation
       return output_data
   ```
3. **Add to main.py:**
   ```python
   import scripts.p0X_new_stage as pX
   output = pX.process_data(input)
   ```

### Script Dependencies

```
p01 ──→ p02 ──→ p03 ──→ p04 ──→ p05 ──→ p06 ──→ p07
parse   preprocess generate train  export inference ocr
```

**Linear execution:** Each stage depends on previous output
**Parallel possible:** For multiple inference runs with same model

---

## Future Azure ML Integration

Each stage can become an **Azure ML Component**:

```python
# Azure ML SDK format
from azure.ai.ml import command_component, input

@command_component(
    inputs={'annotations': input()},
    outputs={'parsed_data': output()}
)
def parse_annotations_component(annotations):
    # Stage code here
```

Components orchestrated in **Azure ML Pipeline**:
- Parameterized paths → Azure Blob Storage
- Distributed training → Azure Compute Clusters
- Model registry → Azure ML Model Registry
- Batch endpoints → Azure ML Endpoints

---

## Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| 1 | XML | Parse | DataFrame |
| 2 | DataFrame | Convert format | YOLO DataFrame |
| 3 | YOLO DF | Split & save | Train/test files |
| 4 | Train files | Train YOLO | best.pt |
| 5 | best.pt | Export | .onnx, .torchscript |
| 6 | Image | Inference | Detections |
| 7 | Detections | OCR | Text |

This modular architecture enables:
- ✅ Development in isolation
- ✅ Testing at each stage
- ✅ Easy debugging
- ✅ Production deployment
- ✅ Cloud migration
