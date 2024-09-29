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

    # Output directory selection
    output_dir = st.text_input("Output Directory", "output")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        st.success(f"Created output directory: {output_dir}")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Create a temporary directory to store the uploaded image
            temp_dir = "temp_streamlit_upload"
            os.makedirs(temp_dir, exist_ok=True)

            # Save the uploaded image
            image_path = os.path.join(temp_dir, uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Process the image
            with st.spinner("Processing image..."):
                result = pipeline.process_image(image_path, output_dir)

            # Display the original image with segmentation
            st.subheader("Segmented Image")
            segmented_image_path = os.path.join(output_dir, "visualizations", f"{result['master_id']}_visualized.jpg")
            st.image(segmented_image_path, use_column_width=True)

            # Display whole image summary
            st.subheader("Whole Image Summary")
            st.write(result['whole_image_summary'])

            # Display extracted text for the whole image
            st.subheader("Extracted Text")
            if result['objects']:
                text_path = os.path.join(output_dir, "extracted_text", f"{result['objects'][0]['object_id']}_text.txt")
                with open(text_path, 'r') as f:
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
            st.write(f"- Segmented Objects: {os.path.abspath(os.path.join(output_dir, 'segmented_objects'))}")
            st.write(f"- Extracted Text: {os.path.abspath(os.path.join(output_dir, 'extracted_text'))}")
            st.write(f"- Summaries: {os.path.abspath(os.path.join(output_dir, 'summaries'))}")
            st.write(f"- Visualizations: {os.path.abspath(os.path.join(output_dir, 'visualizations'))}")
            st.write(f"- Data: {os.path.abspath(os.path.join(output_dir, 'data'))}")

            # Cleanup temporary upload directory
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            st.success(f"Processing complete! Results saved in: {output_dir}")
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
    else:
        st.write("👆 Upload an image to get started!")

# Run the Streamlit app
if __name__ == "__main__":
    run_streamlit_app()
