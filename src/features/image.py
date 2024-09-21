from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from src.models.embeddings.default import model as embedding_model
from src.models.object_detection.default import model as segmentation_model
import cv2


class Mask:
    def __init__(self, bits_mask, box: dict, clss=None) -> None:
        """
        clss: es la clase de la imagen, por ejemplo cat o dog.
        bits_mask: es el array o tensor de bits
        box: limites de la mascara o cuadro segmentado
        """
        self.clss = clss
        self.bits_mask = bits_mask
        self.embedding_attr = None
        self.box = box
        self.center = (
            (box["left"] + box["right"]) / 2,
            (box["top"] + box["bottom"]) / 2,
        )

    def embedding(self):
        """Genera el embedding con un modelo"""
        if self.image is None:
            raise Exception("Image is None")
        if not self.embedding_attr:
            self.embedding_attr = embedding_model.get_image_embedding(self.image)[0]
        return self.embedding_attr


class ImageFeature:
    def __init__(self, image_path, name) -> None:
        super().__init__(),
        self.image = cv2.imread(image_path)
        self.name = name
        self.masks = self.generate_masks()

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return self.name

    def generate_masks(self):
        return segmentation_model.generate_masks(image=self.images)

    def plot_box(self, ax):
        x1, y1 = self.left, self.buttom
        x2, y2 = self.right, self.top

        # Calcular la anchura y altura del rectángulo
        width = x2 - x1
        height = y2 - y1

        color = Color.get_color()
        ax.text(
            self.position[0],
            self.position[1],
            str(self.id),
            ha="center",
            va="center",
            color=color,
        )
        # Añadir el rectángulo a los ejes
        ax.add_patch(Rectangle((x1, y1), width, height, fill=False, edgecolor=color))

    def plot(self):
        if self.image is not None:
            plt.figure(figsize=(8, 8))
            plt.title(f"{self} | pos: {self.position}")
            plt.imshow(self.image)
            plt.axis("off")
            plt.show()
        else:
            print("Image is None")
