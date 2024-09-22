from src.models.embeddings.default import model as embedding_model


class Mask:
    def __init__(self, bits_mask, box: dict, clss=None) -> None:
        """
        clss: es la clase de la imagen, por ejemplo cat o dog.
        bits_mask: es el array o tensor de bits
        box: limites de la mascara o cuadro segmentado
        """
        self.cls = clss
        self.bits_mask = bits_mask
        self.embedding_attr = None
        self.box = box
        self.center = (
            abs(box["left"] + box["right"]) / 2,
            abs(box["top"] + box["bottom"]) / 2,
        )
        self.height = box["top"] - box["bottom"]
        self.width = box["right"] - box["left"]

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

        pass
