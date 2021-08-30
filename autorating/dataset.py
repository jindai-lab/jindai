from typing import Tuple
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class TLDataset(Dataset):
    def __init__(self, images_path: str, transform):
        self.images = glob.glob(images_path + '/*.jpg')
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, np.ndarray]:
        image_path = self.images[item]
        image = default_loader(image_path)
        x = self.transform(image)
        return x, np.array([-1 if os.path.basename(image_path).split('-')[0] == '0' else 1])
