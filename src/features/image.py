# import matplotlib
# matplotlib.use("gtk4agg", force=True)
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from src.models.object_detection.default import model as segmentation_model
from .mask import Mask
import cv2


class ImageFeature:
    def __init__(self, image_path, name) -> None:
        super().__init__(),
        self.image = cv2.imread(image_path)
        self.name = name
        self.masks = []
        self.cls_masks: dict = {}

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return self.name

    def generate_full_masks(self):
        self.masks = segmentation_model.generate_masks(image=self.image)
        return self.masks

    def generate_cls_masks(self, cls):
        self.cls_masks[cls] = segmentation_model.generate_mask_from_cls(
            image=self.image, cls=cls
        )
        return self.cls_masks[cls]

    def fill_mask(self, mask):
        # Hacer el proceso de llenar ese espacio de la mascara
        return mask

    def append_mask(self, mask: Mask, position: tuple, height, width):
        return Exception("Not implemented function")

    def plot_box_from_mask(self, mask: Mask, ax):
        x1 = mask.box["left"]
        y1 = mask.box["bottom"]
        width = mask.width
        height = mask.height

        # color = Color.get_color()
        ax.text(
            mask.cls,
            ha="center",
            va="center",
            # color=color,
        )
        # Añadir el rectángulo a los ejes
        # ax.add_patch(Rectangle((x1, y1), width, height, fill=False, edgecolor=color))
        ax.add_patch(Rectangle((x1, y1), width, height, fill=False))

    def plot(self):
        if self.image is not None:
            plt.figure(figsize=(8, 8))
            plt.title(f"{self.name}")
            plt.imshow(self.image)
            plt.axis("off")
            plt.show()
        else:
            print("Image is None")
