%%writefile app.py
import streamlit as st
import os
import json
from PIL import Image
from pipeline_module import EnhancedMaskRCNNPipeline

# Initialize the pipeline
@st.cache_resource
def load_pipeline():
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weights_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    return EnhancedMaskRCNNPipeline(config_file, weights_file)

# Streamlit app
def run_streamlit_app():
    st.title("AI Pipeline for Image Segmentation and Object Analysis")
    st.write("Upload an image to segment objects, extract text, and analyze!")

    # Load the pipeline
    pipeline = load_pipeline()

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

            # Process the image
            with st.spinner("Processing image..."):
                result = pipeline.process_image(image_path, base_output_dir)

            # Display the segmented image
            st.subheader("Segmented Image")
            segmented_image_path = os.path.join(output_visualizations_dir, f"{result['master_id']}_visualized.jpg")
            st.image(segmented_image_path, use_column_width=True)

            # Display whole image summary
            st.subheader("Whole Image Summary")
            st.write(result['whole_image_summary'])

            # Display extracted text for the whole image
            st.subheader("Extracted Text")
            if result['objects']:
                object_text_path = os.path.join(output_texts_dir, f"{result['objects'][0]['object_id']}_text.txt")
                with open(object_text_path, 'r') as f:
                    extracted_text = f.read().strip()
                    if extracted_text:
                        st.write(extracted_text)
                    else:
                        st.write("No text was extracted from the image.")
            else:
                st.write("No objects detected in the image.")

            # Display object analysis results
            st.subheader("Object Analysis")
            for obj in result['objects']:
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
