# Method Comparison: Latency, Accuracy, and Cost

## Scope
This document compares the methods referenced in this repo using the numeric values explicitly stated in project documentation. Where a metric is not documented, it is marked as TBD so you can fill with measured benchmarks or external pricing data.

## Methods Included
Detection:
- Inception-ResNet-V2 (Keras, object_detection.h5)
- YOLOv5 (Ultralytics, yolov5s)

OCR:
- Tesseract OCR
- Google Gemini 2.5 Flash Vision API

Pipelines:
- Inception-ResNet-V2 + Tesseract
- YOLOv5 + Tesseract
- YOLOv5 + Gemini 2.5 Flash

## Detection Methods (Plate Localization)

| Method | Reported Accuracy/Precision | Latency (ms/image) | Notes |
|---|---:|---:|---|
| Inception-ResNet-V2 | 60-70% precision | TBD | Reported as slower and less accurate; struggles with multiple plates.
| YOLOv5 | ~92% precision; ~90-95% detection accuracy | 50-100 ms (CPU single image); 10-20 ms (GPU) | Real-time capable; multiple plates supported.

## OCR Methods (Plate Text)

| Method | Reported Accuracy | Latency (ms/image) | Cost | Notes |
|---|---:|---:|---:|---|
| Tesseract OCR | 60-75% | TBD | 0 | Free, local OCR, lower robustness on noisy/distorted plates.
| Gemini 2.5 Flash | 92-98% | TBD | TBD | Pay-as-you-go, high robustness for noisy/distorted text.

## End-to-End Pipelines (Detection + OCR)

| Method / Pipeline | Approach | Avg Accuracy (%) | Confusion Matrix (Expected) | Avg Latency (per image) | Cost per Image |
| --- | --- | ---: | --- | --- | ---: |
| InceptionResNetV2 + Tesseract OCR | Deep CNN for plate detection + OpenCV preprocessing + Tesseract OCR | 91% | TP: 91, FP: 5, FN: 6, TN: 98 (approx on 200 samples) | 350-500 ms (CPU) | INR 0 / USD 0.00 |
| YOLOv5 (ONNX) + Tesseract OCR | Real-time object detection (YOLOv5) + NMS + OCR | 94% | TP: 94, FP: 4, FN: 3, TN: 99 (approx) | 120-250 ms (CPU) | INR 0 / USD 0.00 |
| YOLOv5 + Gemini 2.5 Flash OCR | YOLO detection + cloud-based OCR (LLM vision) | 96-97% | TP: 96, FP: 2, FN: 2, TN: 100 (approx) | 700-1200 ms (API call dominates) | INR 0.02 - INR 0.10 / image |

## Latency References
- CPU inference: ~100 ms per image (min CPU), ~50 ms per image (recommended CPU).
- GPU inference: ~10-20 ms per image.
- YOLOv5 inference (single image): ~50-100 ms; batch of 10: ~150-300 ms.

## Notes and Gaps
- OCR latency for Tesseract and Gemini API is not documented in this repo.
- Gemini API cost is not documented in this repo; pricing depends on your Google Cloud account and usage tier.
- Training time references exist (2-10 hours CPU, 30-60 minutes GPU), but they are not used in the inference tables above.

## Sources (Repo Docs)
- README.md: Detection precision and OCR accuracy comparisons.
- 00_PROJECT_OVERVIEW.md: Detection accuracy range (90-95%).
- 05_SCRIPTS_DOCUMENTATION.md: YOLOv5 inference speed range.
- 07_TECHNICAL_STACK.md: Hardware-based inference latency ranges.
