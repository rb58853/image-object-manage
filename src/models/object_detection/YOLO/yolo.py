from ultralytics import YOLO
from ..base import ObjectDetectionModel
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
        raise Exception(
            "YOLO don't need generate all masks. Try use the function `generate_mask_from_cls`"
        )

    def generate_mask_from_cls(self, image, cls: str):
        masks_result = []
        if cls in self.match_names:
            cls_index = self.match_names[cls]
        else:
            raise Exception(f"Yolo can not detect the class {cls}.")

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
            cls_indices = torch.where(clss == cls_index)
            cls_masks = masks[cls_indices]
            cls_boxes = boxes[cls_indices]

            for mask, box in zip(cls_masks, cls_boxes):
                print(mask)
                print(box)
                box = {
                    "left": float(box[0]),
                    "right": float(box[2]),
                    "top": float(box[1]),
                    "bottom": float(box[3]),
                }
                mask = np.array(mask)
                masks_result.append(Mask(bits_mask=mask, box=box, clss=cls))

        return masks_result
