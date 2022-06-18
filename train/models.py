import torch
from torch import nn


class BNLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.activation = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class MatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.player_encoder = nn.Sequential(
            BNLinear(240, 1024, 0.2),
            BNLinear(1024, 1024, 0.2),
            BNLinear(1024, 1024, 0.2),
        )
        self.match_predictor = nn.Sequential(
            BNLinear(2099, 2048, 0.2),
            BNLinear(2048, 2048, 0.2),
            nn.Linear(2048, 1),
            nn.Sigmoid(),
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
