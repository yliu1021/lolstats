import torch
from torch import nn


class MatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.player_encoder = nn.Sequential(
            nn.Linear(160, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU()
        )
        self.match_predictor = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    
    def encode_team(self, team: list[torch.Tensor]) -> torch.Tensor:
        player_embeddings = torch.stack([self.player_encoder(player) for player in team], dim=1)
        team_embedding = torch.sum(player_embeddings, dim=1)
        return team_embedding

    def encode_game(self, game) -> torch.Tensor:
        team1_embeddings = self.encode_team(game["team1"])
        team2_embeddings = self.encode_team(game["team2"])
        return torch.concat([team1_embeddings, team2_embeddings], dim=1)

    def forward(self, game) -> torch.Tensor:
        game_embeddings = self.encode_game(game)
        return self.match_predictor(game_embeddings)
