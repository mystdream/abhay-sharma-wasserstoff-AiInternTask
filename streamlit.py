import streamlit as st
import os
from PIL import Image
from models.segmentation_model import SegmentationModel
from models.text_extraction_model import TextExtractor
from models.summarization_model import Summarizer
from utils.preprocessing import preprocess_image
from utils.visualization import visualize_results
from utils.postprocessing import rle_encode, rle_decode

# Initialize the pipeline
@st.cache_resource
def load_pipeline():
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weights_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    
    segmentation_model = SegmentationModel(config_file, weights_file)
    text_extractor = TextExtractor()
    summarizer = Summarizer()
    
    return segmentation_model, text_extractor, summarizer

# Streamlit app
def run_streamlit_app():
    st.title("AI Pipeline for Image Segmentation and Object Analysis")
    st.write("Upload an image to segment objects, extract text, and analyze!")

    # Load the pipeline
    segmentation_model, text_extractor, summarizer = load_pipeline()

    # Output directory
    base_output_dir = "data"
    input_images_dir = os.path.join(base_output_dir, "input_images")
    segmented_objects_dir = os.path.join(base_output_dir, "segmented_objects", "segmented_objects")
    mapped_files_dir = os.path.join(base_output_dir, "segmented_objects", "mapped_files")
    output_visualizations_dir = os.path.join(base_output_dir, "output", "visualizations")
    output_summaries_dir = os.path.join(base_output_dir, "output", "summaries")
    output_texts_dir = os.path.join(base_output_dir, "output", "extracted_texts")

    # Create directories if they don't exist (before file uploading and processing)
    for directory in [input_images_dir, segmented_objects_dir, mapped_files_dir, output_visualizations_dir, output_summaries_dir, output_texts_dir]:
        os.makedirs(directory, exist_ok=True)

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Save the uploaded image to 'input_images' folder within the output directory
            image_path = os.path.join(input_images_dir, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Preprocess the image
            image = preprocess_image(image_path)

            # Perform object detection
            with st.spinner("Processing image..."):
                outputs = segmentation_model.perform_inference(image)
                results = segmentation_model.postprocess_results(outputs, image)

            # Extract text and summarize the whole image
            full_image_text = text_extractor.extract_text(image_path)
            object_info = [f"{result['class']} (confidence: {result['score']:.2f})" for result in results]
            object_summary = ", ".join(object_info)
            summary_input = f"The image contains the following objects: {object_summary}. The following text was extracted from the image: '{full_image_text}'. Provide a brief summary."
            image_summary = summarizer.summarize(summary_input)

            # Display the segmented image
            st.subheader("Segmented Image")
            master_id = str(uuid.uuid4())
            visualize_results(image, results, segmentation_model.class_names, MetadataCatalog.get(segmentation_model.cfg.DATASETS.TRAIN[0]), output_visualizations_dir, master_id)
            segmented_image_path = os.path.join(output_visualizations_dir, f"{master_id}_visualized.jpg")
            st.image(segmented_image_path, use_column_width=True)

            # Display whole image summary
            st.subheader("Whole Image Summary")
            st.write(image_summary)

            # Display extracted text for the whole image
            st.subheader("Extracted Text")
            st.write(full_image_text if full_image_text else "No text was extracted from the image.")

            # Display object analysis results
            st.subheader("Object Analysis")
            for obj in results:
                with st.expander(f"{obj['class']} (Confidence: {obj['score']:.2f})"):
                    st.image(obj['object_path'], use_column_width=True)

            # Display links to output folders
            st.subheader("Output Folders")
            st.write(f"- Input Images: {os.path.abspath(input_images_dir)}")
            st.write(f"- Segmented Objects: {os.path.abspath(segmented_objects_dir)}")
            st.write(f"- Mapped Files: {os.path.abspath(mapped_files_dir)}")
            st.write(f"- Output (Visualizations): {os.path.abspath(output_visualizations_dir)}")
            st.write(f"- Output (Summaries): {os.path.abspath(output_summaries_dir)}")
            st.write(f"- Output (Extracted Texts): {os.path.abspath(output_texts_dir)}")

            st.success(f"Processing complete! Results saved in: {base_output_dir}")
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
    else:
        st.write("👆 Upload an image to get started!")

# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()
