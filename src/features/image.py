# from PIL import ImageTk
import matplotlib

matplotlib.use("gtk4agg")
# matplotlib.use("TkAgg")

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from src.models.object_detection.default import model as segmentation_model
from .mask import Mask
from src.config.config import Data
import cv2


class ImageFeature:
    """
    ## ImageFeature
    Una ImageFeature es una imagen en si con nombre, y mascaras autogeneradas, las mascaras son objetos detectados con modelos de segmentacion como YOLO o SAM.
    ### inputs
    - **`image_path`**: direccion de una imagen a cargar
    - **`name`**[default = 'image']: nombre que se le dara a la imagen
    ### Attributes
    - **`image`**: Imagen de la ruta cargada con cv2.
    - **`name:str`**: Nombre que se le dio a la instancia de ImageFeature
    - **`masks:List[Mask]`**: Lista de todas las mascaras de la imagen, en un principio es none, en caso de ser utiles se debe pedir la autogeneracion de estas, por lo general se generan todas las mascaras de una vez y luego se accede a estas para crear el diccionario de clases a mascaras.
    - **`cls_masks: dict[str:List[Mask]]`**: Diccionario de `clase(str) -> mascaras(List[Mask])`. Cada key del diccionario representa una clase, por ejemplo `'cat'` y los valores de esta key seran las mascaras que la representan, por ejemplo las mascaras que tienen un gato. Este atributo tambien se autogenera pidiendoselo a la funcion con la clase deseada a buscar.
    ### Functions
    - **`generate_full_masks`**: Genera todas las mascaras de una imagen, es util para generar cada una de las mascaras una sola vez y luego con acceso a estas crear el diccionario de clases a mascara segun las mascaras deseadas.
    """

    def __init__(self, image_path, name="image") -> None:
        super().__init__(),
        self.image = cv2.imread(image_path)
        self.name: str = name
        self.masks: list[Mask] = [] if self.image is None else self.generate_masks()
        self.cls_masks: dict[str : list[Mask]] = {}

    def __str__(self) -> str:
        return f"{self.name}"

    def __repr__(self) -> str:
        return self.name

    def generate_masks(self):
        return segmentation_model.generate_masks(image=self.image)

    def generate_cls_masks(self, cls):
        masks = [mask for mask in self.masks if mask.match_with_cls(cls)]
        self.cls_masks[cls] = masks
        return self.cls_masks[cls]

    def fill_mask(self, mask):
        # Hacer el proceso de llenar ese espacio de la mascara
        return mask

    def append_mask(self, mask: Mask, position: tuple, height, width):
        return Exception("Not implemented function")

    def append_box_from_mask(self, mask: Mask, ax):
        x1 = mask.box["left"]
        y1 = mask.box["bottom"]
        width = mask.width
        height = mask.height

        # color = Color.get_color()
        ax.add_patch(Rectangle((x1, y1), width, height, fill=False, edgecolor="red"))
        # ax.annotate(mask.name, xy= (100,100), xytext=(150,150))
        # ax.add_patch(Rectangle((x1, y1), width, height, fill=False, edgecolor=color))

    def save_xor_plot(self, masks=[], plot=True, save=True):
        path = Data.generation_path(self.name)
        if self.image is not None:
            # plt.figure(figsize=(8, 8))
            fig, ax = plt.subplots()
            ax.imshow(self.image)
            ax.axis("off")
            for mask in masks:
                self.append_box_from_mask(mask=mask, ax=ax)

            plt.title(f"{self.name}")
            if save:
                plt.savefig(path)
            if plot:
                plt.plot()

        else:
            raise Exception("Image is None")
