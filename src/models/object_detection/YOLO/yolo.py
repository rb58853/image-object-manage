from ultralytics import YOLO
import torch


class YoloSegmentation:
    def __init__(self, model_name="yolov8m-seg.pt") -> None:
        self.model = YOLO(model_name)

    def generate_masks(self, image):
        results = self.model.predict(
            source=image.copy(),
            save=False,
            save_txt=False,
            stream=True,
        )
        for result in results:
            # get array results
            masks = result.masks.data
            boxes = result.boxes.data
            # extract classes
            clss = boxes[:, 5]
            # get indices of results where class is 0 (people in COCO)
            cat_indices = torch.where(clss == 15)
            # use these indices to extract the relevant masks
            cat_masks = masks[cat_indices]
            print(cat_masks[0])
            # scale for visualizing results
            cat_masks = torch.any(cat_masks, dim=0).int() * 255
        pass
