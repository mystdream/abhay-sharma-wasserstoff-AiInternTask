# AI Pipeline for Image Segmentation and Object Analysis

## Overview
This project is an **AI-driven image segmentation and analysis pipeline** built using `Detectron2`, `pytesseract`, and `transformers`. The goal is to segment objects in an image, extract text, and generate a summary. The pipeline is integrated with a **Streamlit UI** to allow users to upload images and visualize the results interactively.

## Features
1. **Object Segmentation:** Uses Mask R-CNN to detect and segment objects in the image.
2. **Text Extraction:** Extracts textual content from the image using OCR (`pytesseract`).
3. **Image Summarization:** Uses a summarization model (`BART-large`) to generate a brief summary based on detected objects and extracted text.
4. **Streamlit UI:** A user-friendly interface to upload images, view segmented objects, extracted text, and summaries.

## Installation

### Requirements
- Python 3.8+
- Required Libraries:
  ```bash
  pip install torch torchvision detectron2 pytesseract transformers tqdm opencv-python-headless Pillow streamlit
  ```

### Clone the repository
```bash
git clone https://github.com/your-repo/image-segmentation-pipeline.git
cd image-segmentation-pipeline
```

### Setting up Tesseract OCR
Install Tesseract on your system:
- **Windows:** Download and install from [Tesseract GitHub](https://github.com/tesseract-ocr/tesseract).
- **Linux:** Install via package manager:
  ```bash
  sudo apt install tesseract-ocr
  ```

Make sure `pytesseract.pytesseract.tesseract_cmd` points to the correct installation path.

## How to Run

### Step 1: Run the Streamlit App
```bash
streamlit run app.py
```

### Step 2: Upload an Image
In the Streamlit UI:
1. Upload an image in `.jpg`, `.jpeg`, or `.png` format.
2. The app will process the image, segment objects, extract text, and display the results.

### Outputs
After processing the image, the following outputs are generated:
1. **Segmented Image**: A visual representation with objects detected.
2. **Whole Image Summary**: A summary describing objects and extracted text.
3. **Extracted Text**: The text detected within the image (if any).
4. **Individual Object Images**: Each detected object is saved as a separate image.
5. **Mapped Data (JSON)**: A JSON file mapping object data (classes, bounding boxes, confidence scores) and other relevant metadata.

The results are saved in the following folder structure:
```bash
data/
│
├── input_images/            # Uploaded images
├── segmented_objects/       # Segmented object images and metadata
│   ├── segmented_objects/   # Cropped object images
│   └── mapped_files/        # JSON mappings
├── output/
│   ├── visualizations/      # Segmented visualizations
│   ├── summaries/           # Summarized text
│   └── extracted_texts/     # Extracted text files
```

## Key Functions in the Pipeline

### `preprocess_image(image_path)`
- Loads and preprocesses the image for inference.

### `perform_inference(image)`
- Performs object detection using Mask R-CNN.

### `extract_objects(image, results, segmented_objects_dir)`
- Extracts segmented objects and saves them as separate images.

### `extract_text(image_path)`
- Extracts text using `pytesseract`.

### `summarize_image(results, full_image_text)`
- Generates a summary of the detected objects and extracted text using a summarization model.

### `process_image(image_path, base_output_dir)`
- Processes an image, performs segmentation, text extraction, and saves outputs.

## Troubleshooting
- If there are issues with missing libraries, ensure all dependencies listed in the requirements are installed.
- If `pytesseract` cannot find Tesseract, check the `tesseract_cmd` configuration in your environment.


## Streamlit Application
The application is developed using Streamlit, allowing users to interactively upload images, view segmented objects, and see the confidence scores for object detection. The application outputs the original image with segmentation overlays and a list of detected objects with confidence scores.

- Streamlit UI:

  ![image](streamlit_ui/ui_interface.jpg)

  
- Visual Output:
  
  ![image](data/output/visualizations/e80bb4f7-6f3e-4bc3-8f17-4cd2268c1ba3_visualized.jpg)

  ![image](streamlit_ui/Object_Analysis.jpg)

  ![image](streamlit_ui/output_folder.jpg)
