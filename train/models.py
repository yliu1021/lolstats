import torch
from torch import nn


class MatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(1600, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x) -> torch.Tensor:
        return self.predictor(x)
