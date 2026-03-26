# Automatic Number Plate Recognition (ANPR)

**Powered by YOLOv5 Detection + Pytesseract OCR**

## 🎯 Project Overview

This repository provides a complete **Automatic Number Plate Recognition (ANPR)** system, optimized for local execution. The system first detects the location of a license plate in an image using a high-precision YOLOv5 model and then extracts the alphanumeric characters using the Tesseract OCR engine.

### Highlights
- **End-to-End Pipeline**: Implements both plate detection and character recognition.
- **Fast Inference**: Uses YOLOv5 ONNX models optimized for CPU/GPU via OpenCV DNN.
- **Interactive Testing**: Includes a dedicated Jupyter Notebook (`testing.ipynb`) for real-time image uploads and immediate ANPR feedback.
- **Robust & Extensible**: Supports multiple model formats and provides clear logic for easy modification.

---

## 🏗️ System Architecture: The ANPR Pipeline

The system operates in two distinct stages:

### Stage 1: License Plate Detection
- **Model**: YOLOv5 (ONNX format)
- **Purpose**: To accurately locate the license plate within an input image.
- **Process**:
    1. The image is pre-processed and fed into the YOLOv5 model.
    2. The model returns bounding box coordinates for any detected plates.
    3. Non-Maximum Suppression (NMS) is used to filter for the best possible bounding box.
- **Output**: A precise coordinate box defining the plate's location.

### Stage 2: Text Recognition (OCR)
- **Engine**: Pytesseract (a Python wrapper for Google's Tesseract-OCR Engine).
- **Purpose**: To extract alphanumeric characters from the detected plate region.
- **Process**:
    1. The area defined by the bounding box is cropped from the original image.
    2. The cropped image undergoes preprocessing (grayscale, resizing, thresholding) to improve OCR accuracy.
    3. Pytesseract scans the preprocessed crop and returns the recognized text.
- **Output**: A string containing the license plate number (e.g., "MH20EE7598").

---

## 🚀 Getting Started

### 1. System Dependencies
You must have the Tesseract OCR engine installed on your system.

- **For macOS**:
  ```bash
  brew install tesseract
  ```
- **For Ubuntu/Debian**:
  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr
  ```
- **For Windows**:
  - Download and run the installer from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page.
  - **Important**: Make sure to add the Tesseract installation directory to your system's `PATH` environment variable.

### 2. Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/laxmikhilnani/Automatic-License-Plate-Detection-main.git
cd Automatic-License-Plate-Detection-main
pip install -r requirements.txt
```

### 3. Model Setup

The core models are compressed in `object_detection.h5.zip`. To use them:
1. Open `testing.ipynb`.
2. Run the **"Unzip"** cell (Cell 2) to extract the models into the `results/` folder.
3. The notebook will automatically discover and load the necessary `.onnx` model.

---

## 🛠️ Usage: Interactive ANPR

The easiest way to test the full ANPR pipeline is with [testing.ipynb](testing.ipynb):

1. **Load Models**: Run the initial cells to unzip and load the YOLOv5 model.
2. **Upload & Scan**: Scroll to the final cell, click the **"Upload"** button, and select an image of a vehicle.
3. **View Results**: The system will display the original image with the detected plate number drawn above a green bounding box.

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Plate Detection** | YOLOv5 (Ultralytics) |
| **Text Recognition** | Pytesseract |
| **Model Format** | ONNX |
| **Inference Engine** | OpenCV DNN Module |
| **User Interface** | ipywidgets (Jupyter) |
| **Image Processing** | OpenCV, Pillow |
| **Data Handling** | NumPy, Matplotlib |

---

## 📋 Requirements
- Python 3.8+
- Tesseract OCR Engine (system-installed)
- `opencv-python`
- `pytesseract`
- `tensorflow`
- `torch` (Optional, for `.pt` files)
- `ipywidgets`

---
Developed by [Laxmi Khilnani](https://github.com/laxmikhilnani)
