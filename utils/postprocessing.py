import os
import uuid
import cv2

from .data_mapping import rle_decode

def extract_objects(image, results, segmented_objects_dir):
    for i, result in enumerate(results):
        mask = rle_decode(result['segmentation'], image.shape[:2])
        object_image = image * mask[:, :, np.newaxis]
        object_image[mask == 0] = [255, 255, 255]  # Set background to white
        object_id = str(uuid.uuid4())
        result['object_id'] = object_id
        object_path = os.path.join(segmented_objects_dir, f"{object_id}.png")
        cv2.imwrite(object_path, object_image[:, :, ::-1])
        result['object_path'] = object_path
