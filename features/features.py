from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from models.embeddings.default import model as embedding_model

class Feature:
    def __init__(self) -> None:
        self.items = []
        self.embedding = None
        self.position = None
       
    def __getitem__(self, index):
       return self.items[index]
    
class ImageEmbedding(Feature):
    def __init__(self, image, position) -> None:
        super().__init__()
        self.image = image
        self.image_path = None
        
        self.id = 0
        self.name = 'image'
        self.embedding = None
        self.position = position
        
        self.items = [self.embedding, self.position]
        
        if self.image is not None:
            self.get_embedding()

    def __getitem__(self, index):
       return self.items[index]
    
    def get_embedding(self):
        '''Genera el embedding con un modelo'''
        if self.image is None:
            raise Exception("Image is None")
        self.embedding = embedding_model.get_image_embedding(self.image)[0]
        self.items[0] = self.embedding

    def set_embedding(self, embedding):
        '''Setea un embedding, se usa para extraer el feature desde una lista'''
        self.embedding = embedding
        self.items[0] = self.embedding

    def set_limits(self, limits):
        '''Set in order: left, rigth, top, buttom'''
        self.left, self.right, self.top, self.buttom = limits
    
    def set_id(self, index):
        self.id = index 

    def info(self):
        return f'\
        {self}\n\
        pos: {self.position}\n\
        left: {self.left}\n\
        right: {self.right}\n\
        top: {self.top}\n\
        buttom: {self.buttom}\n\
        '
    
    def plot_region(self,ax):
        x1, y1 = self.left, self.buttom
        x2, y2 = self.right, self.top
        
        # Calcular la anchura y altura del rectángulo
        width = x2 - x1
        height = y2 - y1

        color = Color.get_color()
        ax.text(self.position[0], self.position[1], str(self.id), ha='center', va='center', color=color)
        # Añadir el rectángulo a los ejes
        ax.add_patch(Rectangle((x1, y1), width, height, fill=False, edgecolor=color))

    def __str__(self) -> str:
        return f'{self.name} {self.id}'
    
    def __repr__(self) -> str:
        return f'image {self.id}'

    def plot(self):
        if self.image is not None:
            plt.figure(figsize=(8,8))
            plt.title(f'{self} | pos: {self.position}')
            plt.imshow(self.image)
            plt.axis('off')
            plt.show()
        else:
            print("Image is None")    