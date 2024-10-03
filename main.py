import os
import json
import uuid
from pathlib import Path

# Import the necessary modules
from models.segmentation_model import SegmentationModel
from models.identification_model import IdentificationModel
from models.text_extraction_model import TextExtractionModel
from models.summarization_model import SummarizationModel
from utils.preprocessing import preprocess_image
from utils.postprocessing import extract_objects
from utils.visualization import generate_output

# Set up the necessary directories
base_dir = Path("data")
input_images_dir = base_dir / "input_images"
segmented_objects_dir = base_dir / "segmented_objects"
output_dir = base_dir / "output"

# Create directories if they don't exist
input_images_dir.mkdir(parents=True, exist_ok=True)
segmented_objects_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)

# Define model configurations
CONFIG_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
WEIGHTS_FILE = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
CONFIDENCE_THRESHOLD = 0.7

def process_image(image_path):
    # Step 1: Preprocess the image
    image = preprocess_image(image_path)

    # Step 2: Perform image segmentation
    segmentation_model = SegmentationModel(CONFIG_FILE, WEIGHTS_FILE, CONFIDENCE_THRESHOLD)
    outputs = segmentation_model.perform_inference(image)
    class_names = segmentation_model.get_class_names()

    # Step 3: Post-process segmentation results
    results = IdentificationModel.postprocess_results(outputs, class_names)

    # Step 4: Extract segmented objects
    extract_objects(image, results, segmented_objects_dir)

    # Step 5: Extract text from the image
    full_image_text = TextExtractionModel.extract_text(image_path)

    # Step 6: Summarize image and extracted objects
    summarization_model = SummarizationModel()
    summary = summarization_model.summarize_image(results, full_image_text)

    # Step 7: Generate visualizations
    master_id = str(uuid.uuid4())
    generate_output(image, results, output_dir, master_id, class_names)

    # Step 8: Save the results as JSON
    json_path = output_dir / f"{master_id}_summary.json"
    with open(json_path, "w") as f:
        json.dump({
            "image_summary": summary,
            "objects": results
        }, f, indent=4)

    print(f"Processing complete. Summary saved to {json_path}")

if __name__ == "__main__":
    # Loop through all images in the input_images folder
    for image_file in input_images_dir.iterdir():
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            print(f"Processing image: {image_file.name}")
            process_image(image_file)
        else:
            print(f"Skipping non-image file: {image_file.name}")
