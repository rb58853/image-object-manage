from PIL import Image
from src.models.embeddings.default import model as embedding_model
from src.config.config import Data
from src.config.colors import Color
import numpy as np


class Mask:
    """
    ## Mask
    ### inputs
    - `clss`: es la clase de la imagen, por ejemplo cat o dog.
    - `bits_mask`: es el array o tensor de bits
    - `box`: limites de la mascara o cuadro segmentado
    - `image`: Imagen original desde la cual se formo la mascara
    """

    def __init__(self, bits_mask, box: dict, image: Image, clss=None) -> None:
        self.name = "mask"
        self.cls = clss
        self.bits_mask = bits_mask
        self.embedding_attr = None
        self.box = box
        self.center = (
            abs(box["left"] + box["right"]) / 2,
            abs(box["top"] + box["bottom"]) / 2,
        )
        self.height = abs(box["top"] - box["bottom"])
        self.width = abs(box["right"] - box["left"])

        self.origin_image = image

        self.origin_height = image.height
        self.origin_width = image.width

    def resize_to_origin(self, image, change_bit_mask=True):
        current_width = self.bits_mask.shape[0]
        current_height = self.bits_mask.shape[1]

        scale_factor = max(
            self.origin_width / current_width, self.origin_height / current_height
        )

        new_width = int(current_width * scale_factor)
        new_height = int(current_height * scale_factor)

        # resize image
        resized_img = image.resize((new_width, new_height))

        # Crop image
        x1 = (new_width - self.origin_width) // 2
        x2 = new_width - x1
        y1 = (new_height - self.origin_height) // 2
        y2 = new_height - y1
        resized_img = resized_img.crop((x1, y1, x2, y2))

        

        return resized_img

    def set_name(self, name):
        self.name = name

    def embedding(self):
        """Genera el embedding con un modelo"""
        if self.image is None:
            raise Exception("Image is None")
        if not self.embedding_attr:
            self.embedding_attr = embedding_model.get_image_embedding(self.image)[0]
        return self.embedding_attr

    def match_with_cls(self, cls):
        """
        Devuelve si la mascara matchea con la clase pedida
        """
        if self.cls is not None:
            return self.cls == cls
        else:
            # TODO usar embeddings multimodales para este caso y definir un umbral para el matcheo
            return False

    def resize(self):
        return Exception("Not implemented function")

    def generate_origin_image_mask(self):
        return self.generate_custom_image_mask(
            color=None,
            save=True,
            origin=True,
        )

    def generate_image_mask(self):
        return self.generate_custom_image_mask(
            color=Color()("white"), save=True, tag="_origin"
        )

    def generate_transparent_mask(self, color):
        color = color
        return self.generate_custom_image_mask(
            color=color,
            background_color=(0, 0, 0, 0),
            tag="_transparent",
            save=True,
            opacity=100,
        )

    def generate_custom_image_mask(
        self,
        color=(255, 255, 255),
        background_color=(0, 0, 0, 255),
        tag="",
        save=False,
        opacity=255,
        origin=False,
    ):

        width = self.bits_mask.shape[0]
        height = self.bits_mask.shape[1]
        img = Image.new("RGBA", (width, height), background_color)
        pixels = img.load()

        if origin:
            origin_pixels = self.origin_image.load()

        for row_index, row in enumerate(self.bits_mask):
            for colunm_index, bit in enumerate(row):
                if bit == 1:
                    if not origin:
                        pixels[row_index, colunm_index] = (
                            color[0],
                            color[1],
                            color[2],
                            opacity,
                        )
                    else:
                        pixels[row_index, colunm_index] = origin_pixels[
                            row_index, colunm_index
                        ]

        img = self.resize_to_origin(img)
        if save:
            img.save(f"{Data.generation_path(self.name)}{tag}.png")
        return img