from typing import Any


class FillModel:
    def load_model(self):
        raise Exception("Not implemented function")

    def fill(self, image, mask):
        raise Exception("Not implemented function")

    def __call__(self, image, mask) -> Any:
        return self.fill(image, mask)
