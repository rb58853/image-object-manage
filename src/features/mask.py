from PIL import Image
from src.models.embeddings.default import model as embedding_model
from src.config.config import Data


class Mask:
    def __init__(self, bits_mask, box: dict, image: Image, clss=None) -> None:
        """
        clss: es la clase de la imagen, por ejemplo cat o dog.
        bits_mask: es el array o tensor de bits
        box: limites de la mascara o cuadro segmentado
        """
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

    def resize_to_origin(self, image):
        current_width = self.bits_mask.shape[0]
        current_height = self.bits_mask.shape[1]

        # Calcular el factor de escala
        scale_factor = max(
            self.origin_width / current_width, self.origin_height / current_height
        )

        # Calcular los nuevos tamaños
        new_width = int(current_width * scale_factor)
        new_height = int(current_height * scale_factor)

        # Reducir la imagen
        resized_img = image.resize((new_width, new_height))
        x1 = (new_width - self.origin_width) // 2
        x2 = new_width - x1
        y1 = (new_height - self.origin_height) // 2
        y2 = new_height - y1
        resized_img = resized_img.crop((x1, y1, x2, y2))
        # Guardar la imagen reducida
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

    def generate_image_mask(self):
        width = self.bits_mask.shape[0]
        height = self.bits_mask.shape[1]
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        pixels = img.load()

        for row_index, row in enumerate(self.bits_mask):
            for colunm_index, bit in enumerate(row):
                if bit == 1:
                    pixels[row_index, colunm_index] = (255, 255, 255, 255)

        img.save(f"{Data.generation_path(self.name)}.png")
        return img

    def generate_transparent_mask(self, color=(255, 0, 0)):
        # La mascara se genera al reves, TODO hacer que la mascara llegue como debe ser
        width = self.bits_mask.shape[0]
        height = self.bits_mask.shape[1]
        img = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        pixels = img.load()

        for row_index, row in enumerate(self.bits_mask):
            for colunm_index, bit in enumerate(row):
                if bit == 1:
                    pixels[row_index, colunm_index] = (
                        color[0],
                        color[1],
                        color[2],
                        100,
                    )

        img = self.resize_to_origin(img)
        img.save(f"{Data.generation_path(self.name)}_transparent.png")
        return img
