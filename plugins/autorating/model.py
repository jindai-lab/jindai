import torch
import torch.nn as nn
import torchvision as tv
import os

RESNET18, input_features = tv.models.resnet18, 512

class NIMA(nn.Module):
    def __init__(self, base_model: nn.Module, input_features: int, drop_out: float):
        super(NIMA, self).__init__()
        self.base_model = base_model

        self.head = nn.Sequential(
            nn.ReLU(inplace=True), 
            nn.Dropout(p=drop_out),
            nn.Linear(input_features, 1),
            nn.Softsign()
        )

    def forward(self, x):
        x = self.base_model(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def create_model(drop_out):
    base_model = RESNET18(pretrained=True)
    base_model = nn.Sequential(*list(base_model.children())[:-1])
    model = NIMA(base_model=base_model, input_features=input_features, drop_out=drop_out)
    return model


class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target: torch.Tensor, p_estimate: torch.Tensor):
        assert p_target.shape == p_estimate.shape
        return torch.abs(p_target - p_estimate).mean()
