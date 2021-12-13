import numpy as np
import torch
from torchvision import transforms


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]


class Transform:
    def __init__(self):
        normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

        self._train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize,
            ]
        )

        self._val_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    @property
    def train_transform(self):
        return self._train_transform

    @property
    def val_transform(self):
        return self._val_transform


def set_up_seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

