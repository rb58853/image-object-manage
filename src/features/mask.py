from PIL import Image
from src.models.embeddings.default import model as embedding_model
from src.config.config import Data
from src.config.colors import Color
from src.config.blur import Blur
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

        self.image_manager = MaskImage(self)

    def __call__(self):
        return self.image_manager.object_mask(edge_blur=10, save=False)

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

    def edge(self):
        def is_bit_edge(bits, coor: tuple):
            """
            Funcion para saber si un bit pertenece al borde de la mascara de bits
            """
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

        if self.edge_:
            return self.edge_
        self.edge_ = []
        mask = self.bits_mask
        for i, row in enumerate(mask):
            for j, _ in enumerate(row):
                if is_bit_edge(mask, (i, j)):
                    self.edge_.append((i, j))

        return self.edge_


class MaskImage:
    def __init__(self, mask: Mask):
        self.mask = mask

    def resize(self, image: Image, width, height):
        current_width = image.width
        current_height = image.height

        scale_factor = max(
            width / current_width,
            height / current_height,
        )

        new_width = int(current_width * scale_factor)
        new_height = int(current_height * scale_factor)

        # resize image
        resized_img = image.resize((new_width, new_height))

        # Crop image
        x1 = (new_width - width) / 2
        x2 = new_width - x1
        y1 = (new_height - height) / 2
        y2 = new_height - y1
        resized_img = resized_img.crop((x1, y1, x2, y2))

        return resized_img

    def resize_to_origin(self, image):
        return self.resize(
            image=image, width=self.mask.origin_width, height=self.mask.origin_height
        )

    def object_mask(self, padding=0, edge_blur=0, blur_strong=1, save=True):
        return self.custom_image_mask(
            background_color=(0, 0, 0, 0),
            tag="object_mask",
            save=True,
            padding=padding,
            edge_blur=edge_blur,
            blur_strong=blur_strong,
            opacity=1,
            origin_object=True,
        )


    def image_mask(self, padding=0, edge_blur=0, save=False):
        return self.custom_image_mask(
            color=Color()("white"),
            save=save,
            padding=padding,
            edge_blur=edge_blur,
        )
    
    def image_mask_origin_background(self, padding=0, edge_blur=0, save=False):
        return self.custom_image_mask(
            color=Color()("white"),
            save=save,
            padding=padding,
            edge_blur=edge_blur,
            origin_background=True,
        )

    def transparent_mask(
        self, color=Color()("red"), padding=0, edge_blur=0, save=False
    ):
        color = color
        return self.custom_image_mask(
            color=color,
            background_color=(0, 0, 0, 0),
            tag="_transparent",
            save=save,
            opacity=0.3,
            padding=padding,
            edge_blur=edge_blur,
        )

    def custom_image_mask(
        self,
        color=(255, 255, 255, 255),
        background_color=(0, 0, 0, 255),
        tag="",
        save=False,
        origin_background=False,
        origin_object=False,
        opacity=1,
        padding=0,
        edge_blur=0,
        blur_strong=1,
    ):

        width = self.mask.bits_mask.shape[0]
        height = self.mask.bits_mask.shape[1]

        if origin_background:
            img = self.mask.origin_image.copy()
            img = img.resize((width, height))
        else:
            img = Image.new("RGBA", (width, height), background_color)

        origin_pixels = None
        if origin_object:
            origin_image = self.mask.origin_image.copy()
            origin_image = self.resize(origin_image, width=width, height=height)
            origin_pixels = origin_image.load()

        pixels = img.load()

        for row_index, row in enumerate(self.mask.bits_mask):
            for colunm_index, bit in enumerate(row):
                if bit == 1:
                    if origin_object:
                        origin_pixel = origin_pixels[row_index, colunm_index]
                        pixels[row_index, colunm_index] = (
                            origin_pixel[0],
                            origin_pixel[1],
                            origin_pixel[2],
                            int(opacity * 255),
                        )
                    else:
                        pixels[row_index, colunm_index] = (
                            color[0],
                            color[1],
                            color[2],
                            int(opacity * color[3]),
                        )

        if padding > 0 or edge_blur > 0:
            img = ImageEditor.expand_image(
                image=img,
                edge=self.mask.edge(),
                color=color,
                opacity=opacity,
                edge_blur=edge_blur,
                padding=padding,
                blur_strong=blur_strong,
                background_color=background_color if origin_object else None,
                origin_pixels=origin_pixels,
                fill_with_image=origin_object,
            )

        img = self.resize_to_origin(img)
        if save:
            img.save(f"{Data.generation_path(self.mask.name)}{tag}.png")
        return img

class ImageEditor:
    def expand_image(
        image,
        edge,
        color,
        opacity,
        edge_blur=0,
        padding=0,
        blur_strong=1,
        background_color=None,
        origin_pixels=None,
        fill_with_image=False,
    ):
        """
        `background_color`: Si no es None la imagen se amplia segun el fondo, es decir se pregunta por los pixeles que son iguals al fondo en vez de preguntar por los pixeles que son distintos al color de la mascara. En caso de no ser none entonces tiene que ser el color del fondo
        """
        # Costo n*m encontrar el borde, + sumatoria largo del borde que se autogenera, sumatoria hasta blur+padding

        pixels = image.load()
        new_edge = ImageEditor.edge_expand(
            edge, pixels, mask_color=color, background_color=background_color
        )

        while padding > 0:
            padding -= 1
            for point in new_edge:
                try:
                    if fill_with_image and origin_pixels:
                        origin_pixel = origin_pixels[point[0], point[1]]
                        pixels[point[0], point[1]] = (
                            origin_pixel[0],
                            origin_pixel[1],
                            origin_pixel[2],
                            int(opacity * 255),
                        )
                    else:
                        pixels[point[0], point[1]] = (
                            color[0],
                            color[1],
                            color[2],
                            int(color[3] * opacity),
                        )
                except:
                    continue
            new_edge = ImageEditor.edge_expand(
                new_edge, pixels, mask_color=color, background_color=background_color
            )

        len_blur = edge_blur
        steps = Blur.default_function(len_blur, blur_strong)

        while edge_blur > 0:
            edge_blur -= 1
            opacity = steps.next()

            for point in new_edge:
                try:
                    if fill_with_image and origin_pixels:
                        origin_pixel = origin_pixels[point[0], point[1]]
                        pixels[point[0], point[1]] = (
                            origin_pixel[0],
                            origin_pixel[1],
                            origin_pixel[2],
                            int(opacity * 255),
                        )
                    else:
                        pixels[point[0], point[1]] = (
                            color[0],
                            color[1],
                            color[2],
                            int(color[3] * opacity),
                        )
                except:
                    continue
            new_edge = ImageEditor.edge_expand(
                new_edge, pixels, mask_color=color, background_color=background_color
            )

        return image


    def edge_expand(edge, pixels, mask_color=(255, 255, 255), background_color=None):
        """
        `background_color`: Si es verdadero la imagen se amplia segun el fondo, es decir se pregunta por los pixeles que son iguals al fondo en vez de preguntar por los pixeles que son distintos al color de la mascara.
        """
        # TODO arreglar el tema de supoerposicion para hacer un poco menos de operaciones
        mask_color = (mask_color[0], mask_color[1], mask_color[2])

        def is_mask_pixel(pixel):
            if background_color:
                return (
                    pixel[0] != background_color[0]
                    and pixel[1] != background_color[1]
                    and pixel[2] != background_color[2]
                    and pixel[3] != background_color[3]
                )
            else:
                return (
                    pixel[0] == mask_color[0]
                    and pixel[1] == mask_color[1]
                    and pixel[2] == mask_color[2]
                )

        new_edge = []
        for point in edge:
            for i in [-1, 1]:
                point_ = (point[0] + i, point[1])
                if (
                    ImageEditor.point_in_range(point_, pixels)
                    and point_ not in edge
                    and point_ not in new_edge
                ):
                    pixel = pixels[point_[0], point_[1]]
                    if not is_mask_pixel(pixels[point_[0], point_[1]]):
                        new_edge.append((point_[0], point_[1]))

                point_ = (point[0], point[1] + i)
                if (
                    ImageEditor.point_in_range(point_, pixels)
                    and point_ not in edge
                    and point_ not in new_edge
                ):
                    pixel = pixels[point_[0], point_[1]]
                    if not is_mask_pixel(pixels[point_[0], point_[1]]):
                        new_edge.append((point_[0], point_[1]))
        return new_edge


    def internaEdge():
        pass


    def point_in_range(point, pixels):
        try:
            pixel = pixels[point[0], point[1]]
            return True
        except:
            return False
