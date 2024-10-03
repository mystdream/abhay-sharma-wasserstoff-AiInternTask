%%writefile pipeline_module.py
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

import cv2
import numpy as np
from PIL import Image
import os
import json
import uuid
import easyocr 
from transformers import pipeline
from tqdm import tqdm

class EnhancedMaskRCNNPipeline:
    def __init__(self, config_file, weights_file, confidence_threshold=0.7):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)
        self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes

        # Initialize EasyOCR for text extraction and summarization model
        self.text_extractor = easyocr.Reader(['en']) 
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def perform_inference(self, image):
        with torch.no_grad():
            outputs = self.predictor(image)
        return outputs

    def postprocess_results(self, outputs, image):
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy()

        results = []
        for box, class_id, score, mask in zip(boxes, classes, scores, masks):
            result = {
                "class": self.class_names[class_id],
                "score": float(score),
                "bbox": box.tolist(),
                "segmentation": self.rle_encode(mask)
            }
            results.append(result)

        return results

    def rle_encode(self, mask):
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    def rle_decode(self, rle, shape):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]

        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)

    def extract_objects(self, image, results, segmented_objects_dir):
        for i, result in enumerate(results):
            mask = self.rle_decode(result['segmentation'], image.shape[:2])
            object_image = image * mask[:, :, np.newaxis]
            object_image[mask == 0] = [255, 255, 255]  # Set background to white
            object_id = str(uuid.uuid4())
            result['object_id'] = object_id
            object_path = os.path.join(segmented_objects_dir, f"{object_id}.png")
            cv2.imwrite(object_path, object_image[:, :, ::-1])
            result['object_path'] = object_path

    def extract_text(self, image_path):
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        text_results = self.text_extractor.readtext(image_rgb, detail=0)  # Get only the text
        text = ' '.join(text_results)  # Join detected text into a single string
        return text.strip()

    def summarize_image(self, results, full_image_text):
        object_info = [f"{result['class']} (confidence: {result['score']:.2f})" for result in results]
        object_summary = ", ".join(object_info)

        summary_input = f"The image contains the following objects: {object_summary}. "
        if full_image_text:
            summary_input += f"The following text was extracted from the image: '{full_image_text}'. "
        summary_input += "Provide a brief summary of the image content."

        try:
            summary = self.summarizer(summary_input, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        except Exception as e:
            summary = f"Error generating summary: {str(e)}"

        return summary

    def process_image(self, image_path, base_output_dir):
        master_id = str(uuid.uuid4())
        image = self.preprocess_image(image_path)
        outputs = self.perform_inference(image)
        results = self.postprocess_results(outputs, image)

        # Define subfolder paths
        segmented_objects_dir = os.path.join(base_output_dir, "segmented_objects", "segmented_objects")
        output_visualizations_dir = os.path.join(base_output_dir, "output", "visualizations")
        output_texts_dir = os.path.join(base_output_dir, "output", "extracted_texts")
        output_summaries_dir = os.path.join(base_output_dir, "output", "summaries")
        mapped_files_dir = os.path.join(base_output_dir, "segmented_objects", "mapped_files")

        # Create directories if they don't exist
        for directory in [segmented_objects_dir, output_visualizations_dir, output_texts_dir, output_summaries_dir, mapped_files_dir]:
            os.makedirs(directory, exist_ok=True)

        # Extract text and summarize the whole image
        full_image_text = self.extract_text(image_path)
        image_summary = self.summarize_image(results, full_image_text)

        # Save the whole image summary
        with open(os.path.join(output_summaries_dir, f"{master_id}_whole_image_summary.txt"), 'w') as f:
            f.write(image_summary)

        # Save segmented objects
        self.extract_objects(image, results, segmented_objects_dir)

        # Save extracted text for the whole image
        for result in results:
            result['extracted_text'] = full_image_text
            with open(os.path.join(output_texts_dir, f"{result['object_id']}_text.txt"), 'w') as f:
                f.write(result['extracted_text'])

        # Save results and visualizations
        self.generate_output(image, results, output_visualizations_dir, master_id)

        mapped_data = {
            "master_id": master_id,
            "original_image": image_path,
            "whole_image_summary": image_summary,
            "objects": results
        }

        # Save the mapped file in the mapped_files folder
        with open(os.path.join(mapped_files_dir, f"{master_id}_mapped_data.json"), 'w') as f:
            json.dump(mapped_data, f, indent=2)

        return mapped_data

    def generate_output(self, image, results, output_visualizations_dir, master_id):
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)

        instances = detectron2.structures.Instances(image_size=image.shape[:2])
        instances.pred_boxes = detectron2.structures.Boxes(torch.tensor([r['bbox'] for r in results]))

        instances.scores = torch.tensor([r['score'] for r in results])
        instances.pred_classes = torch.tensor([self.class_names.index(r['class']) for r in results])
        instances.pred_masks = torch.tensor([self.rle_decode(r['segmentation'], image.shape[:2]) for r in results])

        out = v.draw_instance_predictions(instances)
        vis_image = out.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(output_visualizations_dir, f"{master_id}_visualized.jpg"), vis_image)

    def process_directory(self, input_dir, base_output_dir):
        os.makedirs(base_output_dir, exist_ok=True)
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        all_results = []
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(input_dir, image_file)
            result = self.process_image(image_path, base_output_dir)
            all_results.append(result)

        return all_results

if __name__ == "__main__":
    config_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    weights_file = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    pipeline = EnhancedMaskRCNNPipeline(config_file, weights_file)

    input_images_dir = "data/input_images"
    base_output_dir = "data"

    results = pipeline.process_directory(input_images_dir, base_output_dir)
