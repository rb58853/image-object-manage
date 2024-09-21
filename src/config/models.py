from src.models.embeddings.CLIP.clip import ClipEmbedding
from src.models.object_detection.SAM.sam import SAM
from src.models.object_detection.YOLO.yolo import YoloSegmentation


class Models:
    embedding = ClipEmbedding
    object_detection = YoloSegmentation
