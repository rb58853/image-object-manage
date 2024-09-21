from models.embeddings.CLIP.clip import ClipEmbedding
from models.object_detection.SAM.sam import SAM

class Models:
    embedding = ClipEmbedding
    object_detection = SAM

