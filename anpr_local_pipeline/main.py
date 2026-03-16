import os
import logging
import shutil
import pandas as pd # For handling data between steps

import numpy as np # Import numpy
import cv2 # Import cv2

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')

# --- Add scripts directory to sys.path to allow direct imports ---
# This is crucial if main.py is in the project root and scripts are in a subdirectory.
import sys
SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts')
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)
# --- End of sys.path modification ---

# Initialize module aliases to None
parse_annotations = None
preprocess_dataset = None
generate_labels = None
train_model_module = None
export_model_module = None
inference_pipeline_module = None

# Attempt to import script modules
# Script filenames have been changed from '0X_name.py' to 'p0X_name.py' to be valid identifiers.
logging.info("Attempting to import pipeline script modules...")
try:
    import scripts.p01_parse_annotations as parse_annotations
    logging.info("Successfully imported p01_parse_annotations.")
except ImportError as e:
    logging.error(f"Failed to import scripts.p01_parse_annotations: {e}")

try:
    import scripts.p02_preprocess_dataset as preprocess_dataset
    logging.info("Successfully imported p02_preprocess_dataset.")
except ImportError as e:
    logging.error(f"Failed to import scripts.p02_preprocess_dataset: {e}")

try:
    import scripts.p03_generate_labels as generate_labels
    logging.info("Successfully imported p03_generate_labels.")
except ImportError as e:
    logging.error(f"Failed to import scripts.p03_generate_labels: {e}")

# These modules are structure-only due to environment issues and may have internal import problems
try:
    import scripts.p04_train_model as train_model_module
    logging.info("Successfully imported p04_train_model.")
except ImportError as e:
    logging.error(f"Failed to import scripts.p04_train_model: {e}")
    logging.warning("p04_train_model module could not be loaded. Training step will be fully skipped.")

try:
    import scripts.p05_export_model as export_model_module
    logging.info("Successfully imported p05_export_model.")
except ImportError as e:
    logging.error(f"Failed to import scripts.p05_export_model: {e}")
    logging.warning("p05_export_model module could not be loaded. Export step will be fully skipped.")

try:
    import scripts.p06_inference_pipeline as inference_pipeline_module
    logging.info("Successfully imported p06_inference_pipeline.")
except ImportError as e: # This will catch the ModuleNotFoundError for ocr_utils from within p06
    logging.error(f"Failed to import scripts.p06_inference_pipeline: {e}")
    logging.warning("p06_inference_pipeline module could not be loaded. Inference step will be fully skipped.")
    logging.warning("This is likely due to p06 failing to import its own dependencies (e.g., ocr_utils) in this environment.")


# Check if critical data preparation modules were loaded
if not all([parse_annotations, preprocess_dataset, generate_labels]):
    logging.error("One or more critical data preparation script modules (p01, p02, p03) failed to import.")
    sys.exit("Exiting: Essential data preparation scripts could not be loaded.")


# Define project paths (assuming main.py is in project root anpr_local_pipeline)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
ANNOTATIONS_DIR = os.path.join(DATA_DIR, 'annotations')
IMAGES_DIR = os.path.join(DATA_DIR, 'images') # Source images

# Output directories for YOLO data (relative to project root for data.yaml)
YOLO_DATA_IMAGES_BASE = os.path.join(PROJECT_ROOT, 'data_images') # Contains train/test image folders
# data.yaml expects paths relative to where train.py is run (or absolute paths)
# If train.py is run from yolov5/ dir, paths in data.yaml need to be ../data_images/train etc.
# If train.py is run from project root, then data_images/train is fine.
# Our 04_train_model.py is set to run train.py from project_root.

DATA_YAML_PATH = os.path.join(PROJECT_ROOT, 'data.yaml')

# For model training and export (placeholders due to environment issues)
YOLOV5_CLONE_DIR = os.path.join(PROJECT_ROOT, 'yolov5')
TRAINED_MODEL_NAME = 'license_plate_model' # From 04_train_model.py
PLACEHOLDER_BEST_PT_PATH = os.path.join(YOLOV5_CLONE_DIR, 'runs', 'train', TRAINED_MODEL_NAME, 'weights', 'best.pt')
PLACEHOLDER_EXPORTED_ONNX_PATH = os.path.join(YOLOV5_CLONE_DIR, 'runs', 'train', TRAINED_MODEL_NAME, 'weights', 'best.onnx')


# --- Helper function to check for environment issues ---
ENVIRONMENT_WARNING_ISSUED = False
def check_yolov5_environment_blocker():
    global ENVIRONMENT_WARNING_ISSUED
    # This is a conceptual check. The actual error happens during tool execution.
    # We simulate its presence for the main script's logic.
    # If YOLOV5_CLONE_DIR exists and is non-empty, it implies a previous attempt might have partially run.
    # The core issue is that *any* operation touching it can fail in the specific dev env.
    if not ENVIRONMENT_WARNING_ISSUED:
        logging.warning("="*80)
        logging.warning("IMPORTANT: The YOLOv5 training, export, and dependent inference steps")
        logging.warning("are likely to be SKIPPED or use PLACEHOLDERS in this script execution.")
        logging.warning("This is due to persistent environment/tooling errors encountered during")
        logging.warning("development that prevent reliable `git clone` and directory operations")
        logging.warning(f"for the '{YOLOV5_CLONE_DIR}' directory.")
        logging.warning("The script will proceed with structural execution where possible.")
        logging.warning("="*80)
        ENVIRONMENT_WARNING_ISSUED = True
    # For this main script, we assume the blocker is always "active"
    return True


def step_01_parse_annotations():
    logging.info("\n>>> Starting Step 1: Parse Annotations <<<")
    if not os.path.exists(ANNOTATIONS_DIR) or not os.listdir(ANNOTATIONS_DIR):
        logging.error(f"Annotations directory is missing or empty: {ANNOTATIONS_DIR}")
        logging.error("Please populate it with XML annotation files.")
        return None
    try:
        df_annotations = parse_annotations.parsing(ANNOTATIONS_DIR)
        if df_annotations.empty:
            logging.warning("No data parsed from annotations. Check XML files or parsing script.")
            return None
        logging.info(f"Successfully parsed {len(df_annotations)} annotations.")
        # Save to a temporary file or pass in memory
        # For simplicity, let's assume it's passed in memory for now.
        # df_annotations.to_csv(os.path.join(DATA_DIR, 'parsed_annotations.csv'), index=False)
        return df_annotations
    except Exception as e:
        logging.error(f"Error in Step 1 (Parse Annotations): {e}")
        return None

def step_02_preprocess_dataset(df_annotations):
    logging.info("\n>>> Starting Step 2: Preprocess Dataset (Convert to YOLO format) <<<")
    if df_annotations is None or df_annotations.empty:
        logging.error("No annotation data from Step 1 to preprocess.")
        return None
    try:
        df_yolo_format = preprocess_dataset.convert_to_yolo_format(df_annotations)
        if df_yolo_format.empty:
            logging.warning("Preprocessing resulted in an empty dataset. Check class names or data.")
            return None
        logging.info(f"Successfully converted {len(df_yolo_format)} annotations to YOLO format.")
        # df_yolo_format.to_csv(os.path.join(DATA_DIR, 'yolo_formatted_annotations.csv'), index=False)
        return df_yolo_format
    except Exception as e:
        logging.error(f"Error in Step 2 (Preprocess Dataset): {e}")
        return None

def step_03_generate_labels(df_yolo_format):
    logging.info("\n>>> Starting Step 3: Generate Labels and Split Dataset <<<")
    if df_yolo_format is None or df_yolo_format.empty:
        logging.error("No YOLO formatted data from Step 2 to generate labels.")
        return
    if not os.path.exists(IMAGES_DIR) or not os.listdir(IMAGES_DIR):
        logging.error(f"Source images directory is missing or empty: {IMAGES_DIR}")
        logging.error("Please populate it with image files corresponding to annotations.")
        return

    # Clean up previous run's data_images directory
    if os.path.exists(YOLO_DATA_IMAGES_BASE):
        logging.info(f"Removing existing YOLO data directory: {YOLO_DATA_IMAGES_BASE}")
        shutil.rmtree(YOLO_DATA_IMAGES_BASE)

    try:
        # For YOLO, labels are usually in the same directory as images.
        # So, output_image_dir_base and output_label_dir_base are the same.
        generate_labels.generate_yolo_labels(
            df_yolo=df_yolo_format,
            output_image_dir_base=YOLO_DATA_IMAGES_BASE, # e.g., project_root/data_images
            output_label_dir_base=YOLO_DATA_IMAGES_BASE, # For YOLO, labels with images
            source_image_dir=IMAGES_DIR
        )
        logging.info("Successfully generated labels and split dataset.")
        logging.info(f"Train/Test images and labels should be in: {YOLO_DATA_IMAGES_BASE}/train and {YOLO_DATA_IMAGES_BASE}/test")
    except Exception as e:
        logging.error(f"Error in Step 3 (Generate Labels): {e}")

def step_04_train_model():
    logging.info("\n>>> Starting Step 4: Train YOLOv5 Model (Structure Only) <<<")
    if check_yolov5_environment_blocker():
        logging.warning("Skipping actual execution of YOLOv5 training due to environment blocker.")
        logging.info("This step would normally clone YOLOv5, install requirements, and run train.py.")
        logging.info(f"It would use {DATA_YAML_PATH} and save results in {YOLOV5_CLONE_DIR}/runs/train/{TRAINED_MODEL_NAME}")
        logging.info("To simulate progression, we assume a placeholder model 'best.pt' would be created.")

        # Create placeholder directories and dummy best.pt for subsequent steps to "find"
        placeholder_weights_dir = os.path.dirname(PLACEHOLDER_BEST_PT_PATH)
        os.makedirs(placeholder_weights_dir, exist_ok=True)
        if not os.path.exists(PLACEHOLDER_BEST_PT_PATH):
            with open(PLACEHOLDER_BEST_PT_PATH, 'w') as f:
                f.write("This is a dummy best.pt for placeholder purposes.")
            logging.info(f"Created dummy placeholder: {PLACEHOLDER_BEST_PT_PATH}")
        return PLACEHOLDER_BEST_PT_PATH # Return path to placeholder

    # This part would only run if the blocker was not present
    if train_model_module is None:
        logging.error("Train model module was not loaded. Skipping training step.")
        # To allow pipeline to proceed structurally, create dummy placeholder best.pt
        placeholder_weights_dir = os.path.dirname(PLACEHOLDER_BEST_PT_PATH)
        os.makedirs(placeholder_weights_dir, exist_ok=True)
        if not os.path.exists(PLACEHOLDER_BEST_PT_PATH):
            with open(PLACEHOLDER_BEST_PT_PATH, 'w') as f:
                f.write("Dummy best.pt due to train_model_module not loading.")
            logging.info(f"Created dummy placeholder for best.pt: {PLACEHOLDER_BEST_PT_PATH}")
        return PLACEHOLDER_BEST_PT_PATH

    try:
        logging.info("Attempting to run the training module structure...")
        # The train_model_module.main_train() handles cloning, req install, and training.
        # It needs data.yaml and data_images to be ready.
        if not os.path.exists(DATA_YAML_PATH):
            logging.error(f"{DATA_YAML_PATH} not found. Training cannot proceed.")
            return None
        if not os.path.exists(os.path.join(YOLO_DATA_IMAGES_BASE, 'train')) or \
           not os.path.exists(os.path.join(YOLO_DATA_IMAGES_BASE, 'test')):
            logging.error("Training/validation data directories not found under data_images/. Training cannot proceed.")
            return None

        # Call the main function from 04_train_model.py
        # We might want to pass epochs or other params from here.
        # For now, it uses its internal defaults (or 1 epoch if run directly for test).
        train_model_module.main_train(epochs_override=1) # Small epochs for any test run

        # Check for actual best.pt (this would be the real path)
        actual_best_pt = os.path.join(YOLOV5_CLONE_DIR, 'runs', 'train', TRAINED_MODEL_NAME, 'weights', 'best.pt')
        if os.path.exists(actual_best_pt):
            logging.info(f"Training script run (structurally), best model expected at: {actual_best_pt}")
            return actual_best_pt
        else:
            logging.warning(f"Training script run (structurally), but best.pt not found at {actual_best_pt}.")
            logging.warning("Falling back to placeholder path if it exists.")
            return PLACEHOLDER_BEST_PT_PATH if os.path.exists(PLACEHOLDER_BEST_PT_PATH) else None

    except Exception as e:
        logging.error(f"Error in Step 4 (Train Model - Structure): {e}")
        logging.warning("This error might be due to the ongoing environment issues with YOLOv5 setup.")
        return PLACEHOLDER_BEST_PT_PATH if os.path.exists(PLACEHOLDER_BEST_PT_PATH) else None


def step_05_export_model(trained_model_path):
    logging.info("\n>>> Starting Step 5: Export Model (Structure Only) <<<")
    if check_yolov5_environment_blocker():
        logging.warning("Skipping actual execution of YOLOv5 model export due to environment blocker.")
        logging.info(f"This step would normally use {YOLOV5_CLONE_DIR}/export.py.")
        logging.info(f"It would convert {trained_model_path} to ONNX and TorchScript.")

        # Create placeholder exported ONNX file for subsequent steps
        if trained_model_path == PLACEHOLDER_BEST_PT_PATH : # Ensure we use the placeholder ONNX path
            os.makedirs(os.path.dirname(PLACEHOLDER_EXPORTED_ONNX_PATH), exist_ok=True)
            if not os.path.exists(PLACEHOLDER_EXPORTED_ONNX_PATH):
                with open(PLACEHOLDER_EXPORTED_ONNX_PATH, 'w') as f:
                    f.write("This is a dummy best.onnx for placeholder purposes.")
                logging.info(f"Created dummy placeholder: {PLACEHOLDER_EXPORTED_ONNX_PATH}")
            return PLACEHOLDER_EXPORTED_ONNX_PATH # Return path to placeholder ONNX
        else: # Should not happen if blocker is active
            logging.warning("Trained model path is not the placeholder, but blocker is active. This is unexpected.")
            return None


    # This part would only run if the blocker was not present
    if export_model_module is None:
        logging.error("Export model module was not loaded. Skipping export step.")
        # Create dummy placeholder ONNX if not already done by blocker logic
        if trained_model_path == PLACEHOLDER_BEST_PT_PATH :
            os.makedirs(os.path.dirname(PLACEHOLDER_EXPORTED_ONNX_PATH), exist_ok=True)
            if not os.path.exists(PLACEHOLDER_EXPORTED_ONNX_PATH):
                with open(PLACEHOLDER_EXPORTED_ONNX_PATH, 'w') as f:
                    f.write("Dummy best.onnx due to export_model_module not loading.")
                logging.info(f"Created dummy placeholder for best.onnx: {PLACEHOLDER_EXPORTED_ONNX_PATH}")
            return PLACEHOLDER_EXPORTED_ONNX_PATH
        return None # Or a more specific placeholder if needed

    if not trained_model_path or not os.path.exists(trained_model_path):
        logging.error(f"Trained model path ({trained_model_path}) not found. Cannot export.")
        return None
    try:
        logging.info("Attempting to run the export module structure...")
        # The export_model_module.main_export() handles the export call.
        export_model_module.main_export(weights_file_path=trained_model_path)

        # Determine expected ONNX path (this is simplified, export_yolo_model returns a dict)
        exported_onnx_path = trained_model_path.replace('.pt', '.onnx')
        if os.path.exists(exported_onnx_path):
            logging.info(f"Export script run (structurally), ONNX model expected at: {exported_onnx_path}")
            return exported_onnx_path
        else:
            logging.warning(f"Export script run (structurally), but .onnx not found at {exported_onnx_path}.")
            logging.warning("Falling back to placeholder ONNX path if it exists.")
            return PLACEHOLDER_EXPORTED_ONNX_PATH if os.path.exists(PLACEHOLDER_EXPORTED_ONNX_PATH) else None

    except Exception as e:
        logging.error(f"Error in Step 5 (Export Model - Structure): {e}")
        return PLACEHOLDER_EXPORTED_ONNX_PATH if os.path.exists(PLACEHOLDER_EXPORTED_ONNX_PATH) else None

def step_06_inference(onnx_model_path):
    logging.info("\n>>> Starting Step 6: Inference Pipeline (Structure Only) <<<")
    if check_yolov5_environment_blocker(): # Blocker also affects ability to assume model exists
        logging.warning("Inference is dependent on a model, which is affected by environment blocker.")
        # Fall through to use placeholder if available

    if not onnx_model_path or not os.path.exists(onnx_model_path):
        logging.error(f"ONNX model path ({onnx_model_path}) not found. Cannot run inference.")
        logging.warning("Ensure a placeholder ONNX model exists if training/export were skipped.")
        return

    if inference_pipeline_module is None:
        logging.error("Inference pipeline module was not loaded. Skipping inference step.")
        return

    # Use a sample image for inference
    sample_image_path = os.path.join(DATA_DIR, 'sample_test_image.jpg') # User should provide this
    if not os.path.exists(sample_image_path):
        logging.warning(f"Sample image for inference not found: {sample_image_path}")
        logging.info("Creating a dummy black image for inference structure test.")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.imwrite(sample_image_path, dummy_img) # Save it so the script can load it by path

    try:
        logging.info("Attempting to run the inference module structure...")
        # The inference_pipeline_module.main_inference() handles loading model, image, and predicting.
        # It needs the ONNX model path.
        # Check if the inference_pipeline_module has PLACEHOLDER_ONNX_MODEL_PATH attribute
        # This check is more robust than direct comparison if the module failed to load fully.
        if hasattr(inference_pipeline_module, 'PLACEHOLDER_ONNX_MODEL_PATH'):
            if onnx_model_path != inference_pipeline_module.PLACEHOLDER_ONNX_MODEL_PATH:
                logging.warning(f"Main script's ONNX path {onnx_model_path} differs from inference script's default placeholder {inference_pipeline_module.PLACEHOLDER_ONNX_MODEL_PATH}.")
                logging.warning("For structural test, this might indicate an issue if inference script relies on its internal default exclusively.")
                # We could try to set it, but that's too intrusive for a structural demo.
                # For now, just call main_inference and let it use its own default if it's hardcoded to do so.
            inference_pipeline_module.main_inference()
        else:
            logging.warning("inference_pipeline_module does not have PLACEHOLDER_ONNX_MODEL_PATH attribute, calling main_inference without path validation.")
            inference_pipeline_module.main_inference()


        logging.info("Inference script run (structurally). Output image (if any) would be in data/sample_detections_output.jpg")
    except AttributeError as e:
        logging.error(f"Could not find or use parts of inference_pipeline_module: {e}. Ensure it's defined correctly.")
    except Exception as e:
        logging.error(f"Error in Step 6 (Inference - Structure): {e}")


def main():
    logging.info("===== Starting ANPR Local Pipeline (Structure Demonstration) =====")

    # Perform initial environment check to set user expectation
    check_yolov5_environment_blocker()

    # Step 1: Parse Annotations
    df_parsed = step_01_parse_annotations()
    if df_parsed is None:
        logging.error("Halting pipeline: Annotation parsing failed.")
        return

    # Step 2: Preprocess Dataset
    df_yolo = step_02_preprocess_dataset(df_parsed)
    if df_yolo is None:
        logging.error("Halting pipeline: Dataset preprocessing failed.")
        return

    # Step 3: Generate Labels for YOLO
    step_03_generate_labels(df_yolo)
    # This step doesn't return a critical value for the next step's input, but relies on file output.
    # Check if data_images were created (basic check)
    if not os.path.exists(os.path.join(YOLO_DATA_IMAGES_BASE, 'train')):
        logging.error("Halting pipeline: Label generation step seems to have failed (train dir missing).")
        return

    # Step 4: Train Model (Structure Only)
    # This will return PLACEHOLDER_BEST_PT_PATH due to the blocker
    trained_model_pt_path = step_04_train_model()
    if not trained_model_pt_path: # Should get placeholder path
        logging.error("Halting pipeline: Model training step (structure) did not yield a model path.")
        return

    # Step 5: Export Model (Structure Only)
    # This will use trained_model_pt_path (which is a placeholder) and return a placeholder ONNX path
    exported_onnx_model_path = step_05_export_model(trained_model_pt_path)
    if not exported_onnx_model_path: # Should get placeholder path
        logging.error("Halting pipeline: Model export step (structure) did not yield an ONNX model path.")
        return

    # Step 6: Inference (Structure Only)
    # This will use the placeholder ONNX path
    step_06_inference(exported_onnx_model_path)

    logging.info("\n===== ANPR Local Pipeline (Structure Demonstration) Finished =====")
    logging.warning("Reminder: Core ML steps (train, export, inference) were structural placeholders due to environment issues.")

    # Cleanup dummy files created by this main script's placeholder logic
    if os.path.exists(PLACEHOLDER_BEST_PT_PATH) and "dummy best.pt" in open(PLACEHOLDER_BEST_PT_PATH).read():
        os.remove(PLACEHOLDER_BEST_PT_PATH)
        shutil.rmtree(os.path.dirname(PLACEHOLDER_BEST_PT_PATH), ignore_errors=True) # remove .../weights
        shutil.rmtree(os.path.dirname(os.path.dirname(PLACEHOLDER_BEST_PT_PATH)), ignore_errors=True) # remove .../license_plate_model

    if os.path.exists(PLACEHOLDER_EXPORTED_ONNX_PATH) and "dummy best.onnx" in open(PLACEHOLDER_EXPORTED_ONNX_PATH).read():
        os.remove(PLACEHOLDER_EXPORTED_ONNX_PATH)
        # Directories might have been removed by best.pt cleanup if they are shared.
        if os.path.exists(os.path.dirname(PLACEHOLDER_EXPORTED_ONNX_PATH)):
             shutil.rmtree(os.path.dirname(PLACEHOLDER_EXPORTED_ONNX_PATH), ignore_errors=True)


if __name__ == '__main__':
    # Ensure necessary data directories exist or give clear instructions
    if not os.path.exists(ANNOTATIONS_DIR):
        os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
        logging.info(f"Created directory: {ANNOTATIONS_DIR}. Please add your XML annotation files here.")
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR, exist_ok=True)
        logging.info(f"Created directory: {IMAGES_DIR}. Please add your image files here.")

    if not os.listdir(ANNOTATIONS_DIR) or not os.listdir(IMAGES_DIR):
        logging.warning("The 'data/annotations' or 'data/images' directory is empty.")
        logging.warning("The pipeline requires XML annotations and corresponding images to run.")
        logging.info("For a minimal test, you can create dummy files, but results will not be meaningful for data prep steps.")
        # Example dummy files for testing flow:
        # Create a dummy annotation if none exist
        if not os.listdir(ANNOTATIONS_DIR):
            dummy_xml_path = os.path.join(ANNOTATIONS_DIR, 'dummy_annotation.xml')
            with open(dummy_xml_path, 'w') as f:
                f.write('<annotation><filename>dummy_image.jpg</filename><size><width>100</width><height>100</height></size><object><name>license_plate</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>90</xmax><ymax>90</ymax></bndbox></object></annotation>')
            logging.info(f"Created dummy annotation: {dummy_xml_path}")
        # Create a dummy image if none exist
        if not os.listdir(IMAGES_DIR):
            dummy_img_path = os.path.join(IMAGES_DIR, 'dummy_image.jpg') # Must match dummy_annotation.xml
            dummy_img_content = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(dummy_img_path, dummy_img_content)
            logging.info(f"Created dummy image: {dummy_img_path}")

    main()
