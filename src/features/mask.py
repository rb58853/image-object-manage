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
        self.edge_ = None

    def set_name(self, name):
        self.name = name

    def edge(self):
        if self.edge_:
            return self.edge_
        self.edge_ = []
        mask = self.bits_mask
        for i, row in enumerate(mask):
            for j, _ in enumerate(row):
                if is_bit_edge(mask, (i, j)):
                    self.edge_.append((i, j))

        return self.edge_

    def embedding(self):
        """Genera el embedding con un modelo"""
        if self.image is None:
            raise Exception("Image is None")
        if not self.embedding_attr:
            self.embedding_attr = embedding_model.get_image_embedding(self.image)[0]
        return self.embedding_attr

    def __call__(self):
        return self.generate_image_mask(padding=2, save=False)

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
        return
        return self.generate_custom_image_mask(
            color=None,
            save=True,
        )

    def box_mask(self, padding=0, save=True):
        width = self.origin_width
        height = self.origin_height
        img = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        pixels = img.load()

        left = self.box["left"] - padding
        right = self.box["right"] + padding
        top = self.box["top"] - padding
        bottom = self.box["bottom"] + padding

        for row_index, row in enumerate(self.bits_mask):
            for colunm_index, _ in enumerate(row):
                if (
                    colunm_index >= left
                    and colunm_index <= right
                    and row_index >= top
                    and row_index <= bottom
                ):
                    pixels[colunm_index, row_index] = (255, 255, 255, 255)

        if save:
            img.save(f"{Data.generation_path(self.name)}_box.png")
        return img

    def generate_image_mask(self, padding=0, edge_blur=0, save=True):
        return self.generate_custom_image_mask(
            color=Color()("white"),
            save=save,
            padding=padding,
            edge_blur=edge_blur,
        )

    def generate_transparent_mask(self, color, padding=0, edge_blur=0, save=True):
        color = color
        return self.generate_custom_image_mask(
            color=color,
            background_color=(0, 0, 0, 0),
            tag="_transparent",
            save=save,
            opacity=0.3,
            padding=padding,
            edge_blur=edge_blur,
        )

    def generate_custom_image_mask(
        self,
        color=(255, 255, 255, 255),
        background_color=(0, 0, 0, 255),
        tag="",
        save=False,
        opacity=1,
        padding=0,
        edge_blur=0,
    ):

        width = self.bits_mask.shape[0]
        height = self.bits_mask.shape[1]
        img = Image.new("RGBA", (width, height), background_color)
        pixels = img.load()

        for row_index, row in enumerate(self.bits_mask):
            for colunm_index, bit in enumerate(row):
                if bit == 1:
                    pixels[row_index, colunm_index] = (
                        color[0],
                        color[1],
                        color[2],
                        int(opacity * color[3]),
                    )

        if padding > 0 or edge_blur > 0:
            img = expand_image(
                image=img,
                edge=self.edge(),
                color=color,
                opacity=opacity,
                edge_blur=edge_blur,
                padding=padding,
            )

        img = self.resize_to_origin(img)
        if save:
            img.save(f"{Data.generation_path(self.name)}{tag}.png")
        return img


def is_bit_edge(bits, coor: tuple):
    x = coor[0]
    y = coor[1]

    if bits[x][y] == 0:
        return False

    for x_0 in [x - 1, x + 1]:
        try:
            if bits[x_0][y] == 0:
                return True
        except:
            continue

    for y_0 in [y - 1, y + 1]:
        try:
            if bits[x][y_0] == 0:
                return True
        except:
            continue

    return False


def expand_image(image, edge, color, opacity, edge_blur=0, padding=0):
    # Costo n*m encontrar el borde, + sumatoria largo del borde que se autogenera, sumatoria hasta blur+padding
    pixels = image.load()
    new_edge = edge_expand(edge, pixels)

    while padding > 0:
        padding -= 1
        for point in new_edge:
            try:
                pixels[point[0], point[1]] = (
                    color[0],
                    color[1],
                    color[2],
                    color[3] * opacity,
                )
            except:
                continue
        new_edge = edge_expand(new_edge, pixels)

    return image


def edge_expand(edge, pixels):
    # TODO arreglar el tema de supoerposicion para hacer un poco menos de operaciones
    new_edge = []
    for point in edge:
        for i in [-1, 1]:
            if point_in_range((point[0] + i, point[1]), pixels):
                if pixels[point[0] + i, point[1]][0] == 0:
                    new_edge.append((point[0] + i, point[1]))

            if point_in_range((point[0], point[1] + i), pixels):
                if pixels[point[0], point[1] + i][0] == 0:
                    new_edge.append((point[0], point[1] + i))

    return new_edge


def point_in_range(point, pixels):
    try:
        pixel = pixels[point[0], point[1]]
        return True
    except:
        return False
