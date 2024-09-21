from src.config.config import SamEnv as env
from src.features.mask import Mask
import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from ..base import ObjectDetectionModel


class SAM(ObjectDetectionModel):
    """
    ## SAM
    ### Funciones
    - `import_model`: importa el modelo SAM, asigna un valor a `self.mask_generator`
    """

    def __init__(self) -> None:
        self.mask_generator = None
        self.import_model()

    def import_model(self):
        print("PyTorch version:", torch.__version__)
        print("Torchvision version:", torchvision.__version__)
        print("CUDA is available:", torch.cuda.is_available())

        # import sys
        # sys.path.append("..")

        sam_checkpoint = "sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        device = "cuda"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=env.points_per_side,  # 32,
            pred_iou_thresh=env.pred_iou_thresh,  # 0.86,
            stability_score_thresh=env.stability_score_thresh,  # 0.92,
            crop_n_layers=env.crop_n_layers,  # 1,
            crop_n_points_downscale_factor=env.crop_n_points_downscale_factor,  # 2,
            min_mask_region_area=env.min_mask_region_area,  # 20*20,  # Requires open-cv to run post-processing
        )
        return self.mask_generator

    def generate_masks(
        self,
        image,
        min_area=0,
    ):
        """
        ### INPUTS:
        - **`image`**: imagen cargada con cv2
        - **`min_area`**: area minima para un box

        ### OUTPUTS:
        - **`masks`**: Lista de mask generadas
        """
        masks = self.mask_generator.generate(image)
        images_mask = []

        for mask in masks:
            box_im = self.bbox_image(mask["bbox"], image)
            h, w, c = box_im.shape
            box_area = h * w
            if box_area >= min_area:
                bits_mask = mask["segmentation"]
                box = get_box(box_im)

                images_mask.append(Mask(bits_mask=bits_mask, box=box))

        return images_mask


def get_box(bbox):
    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
    return {
        "left": x,
        "right": x + w,
        "bottom": y,
        "top": y + h,
    }
