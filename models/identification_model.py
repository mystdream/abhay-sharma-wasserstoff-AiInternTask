import numpy as np

class IdentificationModel:
    @staticmethod
    def postprocess_results(outputs, class_names):
        instances = outputs["instances"].to("cpu")
        boxes = instances.pred_boxes.tensor.numpy()
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        masks = instances.pred_masks.numpy()

        results = []
        for box, class_id, score, mask in zip(boxes, classes, scores, masks):
            result = {
                "class": class_names[class_id],
                "score": float(score),
                "bbox": box.tolist(),
                "segmentation": IdentificationModel.rle_encode(mask)
            }
            results.append(result)

        return results

    @staticmethod
    def rle_encode(mask):
        pixels = mask.flatten()
        pixels = np.concatenate([[0], pixels, [0]])
        runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
        runs[1::2] -= runs[::2]
        return ' '.join(str(x) for x in runs)

    @staticmethod
    def rle_decode(rle, shape):
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
        return img.reshape(shape)
