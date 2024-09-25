import PIL
import numpy as np
import torch
import os
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image, make_image_grid
from src.config.cuda import check_and_convert_to_cuda
from ..base import FillModel


class Inpainting:
    def __init__(self) -> None:
        self.model = self.load_model()

    def load_model(self):
        pipeline = AutoPipelineForInpainting.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
        )
        pipeline = check_and_convert_to_cuda(pipeline)
        return pipeline

    def fill(self, image, mask, save=True):
        image = image.resize((512, 512))
        mask = mask.resize((512, 512))

        prompt = "fill the mask space using the origin background image environment"
        filled_image = self.model.pipeline(
            prompt=prompt, image=image, mask_image=mask
        ).images[0]

        return filled_image


# unmasked_unchanged_image = pipeline.image_processor.apply_overlay(mask_image, init_image, new_image)
# unmasked_unchanged_image.save("force_unmasked_unchanged.png")
# make_image_grid([init_image, mask_image, repainted_image, unmasked_unchanged_image], rows=2, cols=2)
