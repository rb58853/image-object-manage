from ultralytics import YOLO
from ..base import ObjectDetectionModel
from src.utils.inverse_numpy import inverse_array
from src.features.mask import Mask
import numpy as np
import torch


class YoloSegmentation(ObjectDetectionModel):
    def __init__(self, model_name="yolov8m-seg.pt") -> None:
        self.model = YOLO(model_name)
        self.names = self.model.names
        self.match_names = {
            value: key for key, value in zip(self.names.keys(), self.names.values())
        }

    def generate_masks(self, image):
        masks_dict = {}
        masks_result = []

        predictions = self.model.predict(
            source=image.copy(),
            save=False,
            save_txt=False,
            stream=True,
        )
        for prediction in predictions:
            masks = prediction.masks.data
            boxes = prediction.boxes.data
            clss = boxes[:, 5]
            for cls in clss:
                # masks_dict[cls] = []
                cls_indices = torch.where(clss == cls)
                cls_masks = masks[cls_indices]
                cls_boxes = boxes[cls_indices]

                for mask, box in zip(cls_masks, cls_boxes):
                    box = {
                        "left": float(box[0]),
                        "right": float(box[2]),
                        "top": float(box[1]),
                        "bottom": float(box[3]),
                    }
                    mask = np.array(mask)

                    # hallar el array inverso xq la mascara esta invertida
                    mask = inverse_array(mask)
                    masks_result.append(
                        Mask(
                            bits_mask=mask,
                            box=box,
                            image=image,
                            clss=self.names[int(cls)],
                        )
                    )
                    # masks_dict[cls].append(Mask(bits_mask=mask, box=box, clss=cls))
        return masks_result
        # return masks_dict
