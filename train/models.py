import torch
from torch import nn


class MatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(1600, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.predictor(x)
