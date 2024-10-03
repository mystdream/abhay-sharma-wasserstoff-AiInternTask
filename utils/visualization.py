import os
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

def generate_output(image, results, output_visualizations_dir, master_id, class_names):
    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("coco_2017_train"), scale=1.2)
    
    instances = detectron2.structures.Instances(image_size=image.shape[:2])
    instances.pred_boxes = detectron2.structures.Boxes(torch.tensor([r['bbox'] for r in results]))
    instances.scores = torch.tensor([r['score'] for r in results])
    instances.pred_classes = torch.tensor([class_names.index(r['class']) for r in results])
    instances.pred_masks = torch.tensor([rle_decode(r['segmentation'], image.shape[:2]) for r in results])

    out = v.draw_instance_predictions(instances)
    vis_image = out.get_image()[:, :, ::-1]
    cv2.imwrite(os.path.join(output_visualizations_dir, f"{master_id}_visualized.jpg"), vis_image)
