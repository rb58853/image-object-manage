from models.embeddings.CLIP.clip import ClipEmbedding
from models.object_detection.SAM.sam import SAM
from models.object_detection.YOLO.yolo import YoloSegmentation


class Models:
    embedding = ClipEmbedding
    object_detection = YoloSegmentation
