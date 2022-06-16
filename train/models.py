import torch
from torch import nn


class MatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.player_encoder = nn.Sequential(
            nn.Linear(239, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.match_predictor = nn.Sequential(
            nn.Linear(2099, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1),
            nn.Sigmoid()
        )
    
    def encode_team(self, team: list[torch.Tensor]) -> torch.Tensor:
        team_embedding = sum(self.player_encoder(player) for player in team)
        team_embedding /= len(team)
        return team_embedding

    def encode_game(self, game) -> torch.Tensor:
        team1_embeddings = self.encode_team(game["team1"])
        team2_embeddings = self.encode_team(game["team2"])
        return torch.concat([team1_embeddings, team2_embeddings, game["queue"]], dim=-1)

    def forward(self, game) -> torch.Tensor:
        game_embeddings = self.encode_game(game)
        return self.match_predictor(game_embeddings)
