# Project Structure Documentation

## Complete Directory Tree

```
Automatic-License-Plate-Detection-main/
│
├── 📓 Jupyter Notebook and Scripts (Root Level)
│   ├── automatic-number-plate-recognition-88fa4f-2.ipynb    [Main notebook with complete pipeline]
│   ├── test_load.py                                         [Model loading test utility]
│   ├── inspect_h5.py                                        [H5 model inspection utility]
│   ├── requirements.txt                                     [Python dependencies]
│   ├── data.yaml                                            [YOLOv5 dataset configuration]
│   ├── labels.csv                                           [Extracted label data]
│   └── README.md                                            [Project readme]
│
├── 📂 images/ [Source Annotations - 243 XML files]
│   ├── N1.xml, N2.xml, ..., N248.xml
│   └── [Contains PASCAL VOC format annotations for each image]
│
├── 📂 data_images/ [Processed YOLO Format Data - splits into train/test]
│   ├── train/  [Training dataset - TXT label files]
│   │   ├── N100.txt, N101.txt, ..., N248.txt  [224 training images]
│   │   └── [Each contains normalized YOLO format coordinates]
│   │
│   └── test/   [Testing dataset - TXT label files]
│       ├── N1.txt, N2.txt, ..., N93.txt  [25 test images]
│       └── [Each contains normalized YOLO format coordinates]
│
├── 📂 object_detection/ [TensorFlow Training Logs]
│   ├── train/
│   │   └── events.out.tfevents.1653911242.*.v2  [Training events]
│   │
│   └── validation/
│       └── events.out.tfevents.1653911305.*.v2  [Validation events]
│
├── 📂 results/ [Results and Model Artifacts]
│   ├── object_detection.h5                      [Trained Keras model]
│   ├── labels.csv                               [Results labels]
│   ├── automatic-number-plate-recognition-88fa4f-2.ipynb  [Results notebook]
│   │
│   ├── 📂 object_detection/ [TensorFlow training results]
│   │   ├── train/
│   │   └── validation/
│   │
│   └── 📂 yolov5/ [YOLOv5 Repository Contents]
│       ├── detect.py, train.py, val.py, export.py, hubconf.py
│       ├── requirements.txt
│       ├── models/                [YOLO model architectures]
│       ├── utils/                 [Utility functions]
│       ├── data/                  [Dataset configurations]
│       ├── runs/                  [Training runs]
│       └── [Complete YOLOv5 framework]
│
└── 📂 anpr_local_pipeline/ [Modular ML Pipeline - NEW]
    ├── 📄 main.py                             [Main orchestrator]
    ├── 📄 requirements.txt                    [Pipeline dependencies]
    ├── 📄 data.yaml                           [YOLO configuration]
    ├── 📄 README.md                           [Pipeline documentation]
    │
    ├── 📂 data/ [Raw data directory]
    │   ├── images/                            [Original images folder (empty - user populated)]
    │   ├── annotations/                       [XML annotation folder]
    │   │   └── dummy_annotation.xml           [Placeholder annotation]
    │   └── data_images/                       [Generated train/test splits]
    │       ├── train/
    │       └── test/
    │
    ├── 📂 scripts/ [Processing Pipeline Scripts]
    │   ├── __pycache__/                       [Python cache]
    │   ├── 07_ocr_utils.py                    [OCR utility functions]
    │   ├── p01_parse_annotations.py           [Stage 1: Parse XML annotations]
    │   ├── p02_preprocess_dataset.py          [Stage 2: Convert to YOLO format]
    │   ├── p03_generate_labels.py             [Stage 3: Generate labels and split]
    │   ├── p04_train_model.py                 [Stage 4: Train YOLO model]
    │   ├── p05_export_model.py                [Stage 5: Export to ONNX/TorchScript]
    │   └── p06_inference_pipeline.py          [Stage 6: Run inference]
    │
    └── 📂 data_images/ [Generated data (created after running pipeline)]
        ├── train/
        │   └── dummy_image.txt                [Placeholder label file]
        └── test/

└── 📂 Docs/ [NEW - Comprehensive Documentation]
    ├── 00_PROJECT_OVERVIEW.md                 [This file - project overview]
    ├── 01_PROJECT_STRUCTURE.md                [Directory organization]
    ├── 02_ROOT_FILES.md                       [Root level files]
    ├── 03_DATA_FILES.md                       [Data format and structure]
    ├── 04_ANPR_PIPELINE.md                    [Pipeline architecture]
    ├── 05_SCRIPTS_DOCUMENTATION.md            [Script details]
    ├── 06_NOTEBOOK_DOCUMENTATION.md           [Notebook contents]
    └── 07_TECHNICAL_STACK.md                  [Technology details]
```

---

## Directory Descriptions

### Root Level (`/`)
The root directory contains the main implementation and project-wide resources.

**Key Files:**
- `automatic-number-plate-recognition-88fa4f-2.ipynb` - Complete end-to-end notebook
- `requirements.txt` - All Python dependencies
- `data.yaml` - YOLOv5 dataset configuration
- `labels.csv` - Processed label data

### images/ 
Contains 243 XML annotation files in PASCAL VOC format.

**Contents:**
- Named sequentially: N1.xml, N2.xml, ..., N248.xml
- Each file describes one image with bounding boxes
- Organized by image number ID

### data_images/ 
Processed dataset split into train/test for YOLO training.

**Train Folder:**
- 224 .txt label files
- YOLO normalized format coordinates
- Sequential naming: N100.txt, N101.txt, etc.

**Test Folder:**
- 25 .txt label files
- YOLO format coordinates
- Used for validation/testing

### object_detection/
TensorFlow event logs from model training.

**Structure:**
```
object_detection/
├── train/ → TensorFlow training events
└── validation/ → Validation events
```

### results/
Final outputs and model artifacts.

**Contents:**
- `object_detection.h5` - Final trained model
- `labels.csv` - Processed labels
- `object_detection/` - Training logs
- `yolov5/` - Complete YOLOv5 clone

### anpr_local_pipeline/
Modular, production-ready pipeline implementation.

**Sub-directories:**

#### scripts/
Seven processing stages:
1. `p01_parse_annotations.py` - XML parsing
2. `p02_preprocess_dataset.py` - YOLO conversion
3. `p03_generate_labels.py` - Data splitting
4. `p04_train_model.py` - Model training
5. `p05_export_model.py` - Model export
6. `p06_inference_pipeline.py` - Inference
7. `07_ocr_utils.py` - OCR functions

#### data/
- `images/` - Raw input images (user-populated)
- `annotations/` - XML annotation files
- `data_images/` - Generated processed data

---

## File Organization Principles

### 1. Separation of Concerns
- Each script handles one pipeline stage
- Clear input/output contracts
- Independent execution possible

### 2. Data Flow
```
images/NAnnnotations → p01: Parse
                            ↓
                      p02: Preprocess to YOLO
                            ↓
                      p03: Generate labels
                            ↓
                      data_images/[train, test]
                            ↓
                      p04: Train model
                            ↓
                      p05: Export model
                            ↓
                      p06: Run inference
                            ↓
                      p07: OCR extraction
```

### 3. Configuration Management
- `data.yaml` - Centralized dataset config
- Script arguments for customization
- Logging for debugging

### 4. Model Storage
- `.h5` files - Keras format
- `.pt` files - PyTorch format
- `.onnx` - Export format
- Results folder for artifacts

---

## Size and Scale

| Component | Count | Size |
|-----------|-------|------|
| Source Images | 243 | XML files |
| Training Samples | 224 | YOLO .txt files |
| Test Samples | 25 | YOLO .txt files |
| Scripts | 7 | Python files |
| Models | 1 (primary) | .h5 format |
| Documentation Files | 8 | Markdown |

---

## Access Patterns

### For Training
```
anpr_local_pipeline/
├── data/annotations/ → p01 → p02 → p03
├── p04_train_model.py
└── data.yaml
```

### For Inference
```
anpr_local_pipeline/
├── p06_inference_pipeline.py
├── 07_ocr_utils.py
└── [saved model files]
```

### For Analysis
```
results/
├── object_detection.h5
├── object_detection/
└── labels.csv
```

---

## Workflow Integration

### Development Workflow
1. Modify data in `images/` and `anpr_local_pipeline/data/`
2. Run pipeline scripts in order
3. Check outputs in `data_images/`
4. Train model with `p04_train_model.py`
5. View results in `results/`

### Production Workflow
1. Use pre-trained model from `results/`
2. Place input images in specific directory
3. Run inference with `p06_inference_pipeline.py`
4. Extract plates with `07_ocr_utils.py`
5. Return results

### Deployment Workflow
1. Export model with `p05_export_model.py`
2. Deploy to Azure ML / Cloud
3. Use Flask API wrapper
4. Scale horizontally with containers

---

## Notes on File Organization

- **Naming Convention:** Sequential image numbering (N#) maintained across all directories
- **Format Standardization:** A single XM format for annotations, YOLO format for training
- **Backward Compatibility:** Root-level files compatible with original project
- **Forward Compatibility:** anpr_local_pipeline designed for cloud migration
