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
        self.cfg.MODEL.DEVICE = "cpu"
        self.predictor = DefaultPredictor(self.cfg)
        self.class_names = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]).thing_classes

    def perform_inference(self, image):
        with torch.no_grad():
            outputs = self.predictor(image)
        return outputs
