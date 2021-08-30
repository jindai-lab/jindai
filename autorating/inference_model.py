from pathlib import Path

import torch
from PIL.Image import Image
from torchvision.datasets.folder import default_loader

from .common import Transform
from .model import create_model


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def load_state(path_to_model_state):
    return torch.load(path_to_model_state, map_location=lambda storage, loc: storage)


class InferenceModel:
    def __init__(self, model_state=None, path_to_model_state=Path('best.pth')):
        self.transform = Transform().val_transform
        if model_state is None:
            model_state = load_state(path_to_model_state)
        self.model = create_model(drop_out=0)
        self.model.load_state_dict(model_state["state_dict"])
        self.model = self.model.to(device)
        self.model.eval()

    def predict_from_file(self, image_path: Path):
        image = default_loader(image_path)
        return self.predict(image)

    def predict_from_pil_image(self, image: Image):
        image = image.convert("RGB")
        return self.predict(image)

    @torch.no_grad()
    def predict(self, image):
        image = self.transform(image)
        image = image.unsqueeze_(0)
        image = image.to(device)
        prob = self.model(image).data.cpu()[0]
        del image
        if use_cuda:
            torch.cuda.empty_cache()
        return prob
    