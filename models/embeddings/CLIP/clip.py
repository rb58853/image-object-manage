import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ..base import EmbeddingModel


class ClipEmbedding(EmbeddingModel):
    def __init__(self) -> None:
        # self.model, self.processor, self.device = (None,None,None)
        self.model, self.processor, self.device = self.get_model()

    def get_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return (model, processor, device)

    def get_image_embedding(self, image):
        image_process = self.processor(text=None, images=image, return_tensors="pt")[
            "pixel_values"
        ].to(self.device)

        embedding = self.model.get_image_features(image_process)
        # return embedding

        # si quiere usarse numpy arrays cometar el return
        embedding_as_np = embedding.cpu().detach().numpy()
        return embedding_as_np

    def get_text_embedding(self, text):
        image = Image.new("RGB", (10, 10), color=(0, 0, 0))
        if not isinstance(text, list):
            text = [text]
        encoded_text = self.processor(
            text=text,
            images=image,
            padding=True,
            # truncation=True,
            # max_length=100,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**encoded_text)
        text_embeds = outputs["text_embeds"]
        # return text_embeds

        # si quiere usarse numpy arrays cometar el return
        embedding_as_np = text_embeds.cpu().detach().numpy()
        return embedding_as_np
