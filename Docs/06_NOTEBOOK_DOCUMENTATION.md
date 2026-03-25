# Jupyter Notebook Documentation

## Overview

The main Jupyter notebook `automatic-number-plate-recognition-88fa4f-2.ipynb` is a comprehensive, executable walkthrough of the entire license plate detection and recognition pipeline.

---

## Notebook Summary

**File Name:** `automatic-number-plate-recognition-88fa4f-2.ipynb`
**Total Cells:** 107 cells (mix of Markdown, Code, and Output)
**Execution Status:** Not executed in current state
**Framework:** TensorFlow/Keras 2.11.0

---

## Cell-by-Cell Breakdown

### Section 1: Introduction & Setup (Cells 1-10)

#### Cell 1: Title and Overview (Markdown)
- Project title and description
- Key objectives
- Expected outcomes

#### Cells 2-5: Library Imports (Code)
```python
import tensorflow as tf
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
```

**Purpose:**
- TensorFlow: Deep learning framework
- OpenCV: Computer vision
- Pandas: Data manipulation
- NumPy: Numerical computing
- Matplotlib/Seaborn: Visualization
- sklearn: Machine learning utilities

#### Cells 6-10: Configuration (Code)
- Set data paths
- Define constants (image size, batch size, epochs)
- Initialize random seeds for reproducibility
- Configure GPU/TPU settings (if available)

---

### Section 2: Data Loading & Exploration (Cells 11-25)

#### Cell 11: Load CSV Labels (Code)
```python
df = pd.read_csv('labels.csv')
print(df.head())
print(df.shape)
```

**Loads:**
- CSV file with bounding box annotations
- Columns: filepath, xmin, xmax, ymin, ymax

**Output:**
- DataFrame with 400+ rows
- Shows first few annotations

#### Cells 12-15: Data Statistics (Code)
```python
# Calculate statistics
df['width'] = df['xmax'] - df['xmin']
df['height'] = df['ymax'] - df['ymin']
df['area'] = df['width'] * df['height']

print(df.describe())
print(f"Average plate size: {df['width'].mean():.2f} x {df['height'].mean():.2f}")
```

**Outputs:**
- Bounding box statistics
- Width/height distributions
- Area statistics
- Min/max values

#### Cells 16-20: Data Visualization (Code)
```python
# Histogram of plate widths
plt.hist(df['width'], bins=50, edgecolor='black')
plt.title('Distribution of License Plate Widths')
plt.xlabel('Width (pixels)')
plt.ylabel('Count')
plt.show()

# Similar for heights and areas
```

**Visualizations:**
- Width distribution histogram
- Height distribution histogram
- Area distribution
- Box plot comparisons

#### Cells 21-25: Data Quality Check (Code)
- Verify no missing values
- Check coordinate ranges
- Validate image references
- Identify outliers

---

### Section 3: Data Preprocessing (Cells 26-40)

#### Cell 26: PASCAL VOC to YOLO Conversion (Code)
```python
def convert_to_yolo_format(xmin, ymin, xmax, ymax, img_width, img_height):
    """Convert PASCAL VOC to YOLO format"""
    center_x = (xmin + xmax) / 2 / img_width
    center_y = (ymin + ymax) / 2 / img_height
    width = (xmax - xmin) / img_width
    height = (ymax - ymin) / img_height
    return center_x, center_y, width, height
```

**Formula Explained:**
- Calculates normalized center coordinates
- Normalizes width and height

#### Cells 27-30: Apply Conversion (Code)
```python
# Apply conversion to all rows
df['center_x'] = df.apply(lambda row: ..., axis=1)
df['center_y'] = df.apply(lambda row: ..., axis=1)
df['width_norm'] = df.apply(lambda row: ..., axis=1)
df['height_norm'] = df.apply(lambda row: ..., axis=1)

print(df[['center_x', 'center_y', 'width_norm', 'height_norm']].head())
```

**Results:**
- New normalized coordinate columns
- Values between 0 and 1
- Ready for YOLO format

#### Cells 31-35: Validation (Code)
- Verify all values in [0, 1] range
- Check for NaN or invalid values
- Display sample conversions

#### Cells 36-40: Visualization (Code)
- Plot normalized coordinates
- Show box distributions in normalized space
- Verify conversion correctness

---

### Section 4: Dataset Splitting (Cells 41-50)

#### Cell 41: Train/Test Split (Code)
```python
from sklearn.model_selection import train_test_split

# Get unique filenames
filenames = df['filename'].unique()

# Split 80-20
train_files, test_files = train_test_split(
    filenames,
    test_size=0.2,
    random_state=42
)

print(f"Train: {len(train_files)}, Test: {len(test_files)}")
```

**Result:**
- ~224 training images (80%)
- ~49 test images (20%)
- Reproducible split (random_state=42)

#### Cells 42-45: Create Directory Structure (Code)
```python
import os
import shutil

os.makedirs('data_images/train', exist_ok=True)
os.makedirs('data_images/test', exist_ok=True)

# Copy images to folders
for file in train_files:
    src = f'images/{file}'
    dst = f'data_images/train/{file}'
    shutil.copy(src, dst)
```

**Creates:**
```
data_images/
├── train/ (224 images)
└── test/ (49 images)
```

#### Cells 46-48: Save YOLO Labels (Code)
```python
# Save labels for YOLO training
for filename in train_files:
    # Get annotations for this image
    annotations = df[df['filename'] == filename]
    
    # Write YOLO format
    with open(f'data_images/train/{filename[:-4]}.txt', 'w') as f:
        for _, row in annotations.iterrows():
            f.write(f"{0} {row['center_x']} {row['center_y']} {row['width_norm']} {row['height_norm']}\n")
```

**Output Files:**
- `data_images/train/*.txt` (224 files)
- `data_images/test/*.txt` (49 files)

#### Cells 49-50: Verification (Code)
- Count created files
- Sample file contents
- Verify split integrity

---

### Section 5: Model Architecture (Cells 51-65)

#### Cell 51: Define Model (Code)
```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, 3, activation='relu', input_shape=(640, 640, 3)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    # ... more layers
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary output
])
```

**Architecture:**
- Convolutional layers for feature extraction
- Max pooling for downsampling
- Dense layers for classification
- YOLO detection head

#### Cells 52-55: Model Summary (Code)
```python
model.summary()
print(f"Total parameters: {model.count_params():,}")

# Plot architecture
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_architecture.png', show_shapes=True)
```

**Output:**
- Layer-by-layer summary
- Parameter counts
- Model visualization

#### Cells 56-60: Load Pre-trained Weights (Code)
```python
# Load pre-trained YOLO weights
model.load_weights('object_detection.h5')

# Freeze some layers for transfer learning
for layer in model.layers[:-10]:
    layer.trainable = False
```

**Strategy:**
- Transfer learning from pre-trained model
- Freeze early layers (general features)
- Fine-tune later layers (specific features)

#### Cells 61-65: Compilation (Code)
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)
```

**Configuration:**
- Optimizer: Adam (adaptive learning rate)
- Loss: Binary crossentropy (binary classification)
- Metrics: Accuracy, precision, recall

---

### Section 6: Training (Cells 66-80)

#### Cells 66-70: Data Generators (Code)
```python
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    'data_images/train',
    target_size=(640, 640),
    batch_size=8
)
```

**Augmentation:**
- Rotation ±10°
- Shift ±10%
- Zoom ±20%
- Horizontal flip

#### Cells 71-75: Training (Code)
```python
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=val_generator,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
)
```

**Configuration:**
- 100 epochs
- Early stopping if no improvement for 10 epochs
- Save best model

**Output:**
- Training history (loss, accuracy over epochs)
- Best model checkpoint

#### Cells 76-80: Training Visualization (Code)
```python
# Plot training curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss Over Epochs')
plt.show()

# Accuracy curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
```

**Visualizations:**
- Loss curve (should decrease)
- Accuracy curve (should increase)
- Validation performance
- Overfitting detection

---

### Section 7: Evaluation (Cells 81-95)

#### Cells 81-85: Test Set Performance (Code)
```python
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Metrics
predictions = model.predict(test_generator)
y_pred = (predictions > 0.5).astype(int)
y_true = test_generator.classes

from sklearn.metrics import precision_score, recall_score, f1_score
print(f"Precision: {precision_score(y_true, y_pred):.4f}")
print(f"Recall: {recall_score(y_true, y_pred):.4f}")
print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
```

**Metrics:**
- Test loss and accuracy
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1-Score: Harmonic mean

#### Cells 86-90: Confusion Matrix (Code)
```python
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred))
```

**Output:**
- Confusion matrix heatmap
- Classification report
- Per-class metrics

#### Cells 91-95: ROC Curve (Code)
```python
from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(y_true, predictions)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

**ROC Curve:**
- Plot of TPR vs FPR
- AUC (Area Under Curve) metric
- Model discrimination ability

---

### Section 8: Inference & Detection (Cells 96-101)

#### Cell 96: Load Test Image (Code)
```python
import cv2

test_image_path = 'data_images/test/sample.jpg'
image = cv2.imread(test_image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.title('Test Image')
plt.show()
```

#### Cells 97-99: Preprocess & Inference (Code)
```python
# Preprocess
image_resized = cv2.resize(image, (640, 640))
image_normalized = image_resized / 255.0
image_batch = image_normalized[np.newaxis, ...]

# Inference
predictions = model.predict(image_batch)
```

#### Cell 100: Draw Predictions (Code)
```python
def draw_boxes(image, predictions, threshold=0.5):
    """Draw bounding boxes on image"""
    
    # Get detections above threshold
    detections = predictions[predictions['confidence'] > threshold]
    
    for _, det in detections.iterrows():
        x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
        conf = det['confidence']
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        text = f"License Plate: {conf:.2%}"
        cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, (0, 255, 0), 2)
    
    return image

result_image = draw_boxes(image.copy(), detections)
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Detected License Plates')
plt.show()
```

#### Cell 101: OCR Integration (Code - Optional)
```python
# Extract text from detected plates
import pytesseract

for _, det in detections.iterrows():
    x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
    
    # Crop plate region
    plate = image[y1:y2, x1:x2]
    
    # Extract text
    text = pytesseract.image_to_string(plate)
    print(f"Plate text: {text}")
```

---

## Key Notebook Features

### 1. Modular Structure
- Clear section separations
- Easy to skip/restart sections
- Reusable functions

### 2. Comprehensive Visualization
- Histograms and distributions
- Training curves
- Confusion matrices
- ROC curves
- Sample predictions

### 3. Error Handling
- Try-except blocks
- Informative error messages
- Data validation checks

### 4. Documentation
- Markdown cell explanations
- Code comments
- Output interpretation

### 5. Reproducibility
- Fixed random seeds
- Documented parameters
- Version specifications

### 6. Save Outputs
- Model checkpoints
- Predictions
- Visualizations
- Metrics logs

---

## Notebook Execution Flow

```
1. Import & Setup
    ↓
2. Load & Explore Data
    ↓
3. Preprocess Data
    ↓
4. Split Dataset
    ↓
5. Build Model
    ↓
6. Train Model
    ↓
7. Evaluate Model
    ↓
8. Run Inference
    ↓
9. Visualize Results
    ↓
10. Save Model & Results
```

---

## Python Versions & Dependencies

**Minimum Python:** 3.8
**Tested with:** Python 3.9+

**Key Packages:**
```
tensorflow==2.11.0
opencv-python
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## Running the Notebook

### Option 1: Jupyter Notebook UI
```bash
jupyter notebook automatic-number-plate-recognition-88fa4f-2.ipynb
```

### Option 2: Jupyter Lab
```bash
jupyter lab automatic-number-plate-recognition-88fa4f-2.ipynb
```

### Option 3: Command Line Execution
```bash
jupyter nbconvert --to notebook --execute automatic-number-plate-recognition-88fa4f-2.ipynb
```

---

## Expected Output

After execution, expect:
1. Data statistics and visualizations
2. Model architecture summary
3. Training curves (loss, accuracy)
4. Evaluation metrics (precision, recall, F1)
5. Confusion matrix
6. ROC curve with AUC
7. Sample predictions with visualizations
8. Saved model: `object_detection.h5`

---

## Customization Notes

**Parameters to modify:**
- `test_size`: Train/test split ratio
- `epochs`: Number of training iterations
- `batch_size`: Samples per batch
- `image_size`: Input image dimensions
- `augmentation`: Data augmentation parameters

**Model modification:**
- Add/remove layers
- Adjust layer sizes
- Change activation functions
- Modify optimizer settings

**Data modification:**
- Use different dataset
- Add more preprocessing steps
- Apply different augmentations

---

## Troubleshooting

**Out of Memory:**
- Reduce batch size
- Reduce image size
- Use gradient checkpointing
- Clear notebook outputs

**Slow Training:**
- Use GPU (CUDA)
- Reduce number of layers
- Use smaller model architecture
- Reduce dataset size for testing

**Low Accuracy:**
- Increase training data
- Increase epochs
- Improve data augmentation
- Adjust learning rate
- Fine-tune hyperparameters

---

## Notes

- Notebook is comprehensive but can be run section-by-section
- Each section is relatively independent
- Outputs shown in cells may differ based on data and environment
- GPU access recommended for faster training
- Results may vary slightly due to randomization
