# models/segmentation_model.py
import torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog

class SegmentationModel:
    def __init__(self, config_file, weights_file, confidence_threshold=0.7):
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(config_file))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(weights_file)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        self.cfg.MODEL.DEVICE = "cpu"  # Use "cuda" if GPU is available
        self.predictor = DefaultPredictor(self.cfg)
        self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes

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
