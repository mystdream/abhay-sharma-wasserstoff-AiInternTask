# utils/visualization.py
import os
import cv2
from detectron2.utils.visualizer import Visualizer
import detectron2

def visualize_results(image, results, class_names, metadata, output_dir, master_id):
    v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
    
    instances = detectron2.structures.Instances(image_size=image.shape[:2])
    instances.pred_boxes = detectron2.structures.Boxes(torch.tensor([r['bbox'] for r in results]))
    instances.scores = torch.tensor([r['score'] for r in results])
    instances.pred_classes = torch.tensor([class_names.index(r['class']) for r in results])
    instances.pred_masks = torch.tensor([rle_decode(r['segmentation'], image.shape[:2]) for r in results])

    out = v.draw_instance_predictions(instances)
    vis_image = out.get_image()[:, :, ::-1]
    cv2.imwrite(os.path.join(output_dir, f"{master_id}_visualized.jpg"), vis_image)

def rle_decode(rle, shape):
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
