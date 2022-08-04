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


class Embedding(nn.Module):
    def __init__(self, num_indices: int, embedding_dims: int, num_ones: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(embedding_dims, num_indices))
        nn.init.normal_(self.weight, mean=0, std=1 / num_ones)

    def forward(self, x):
        return nn.functional.linear(x, self.weight)


class MatchModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.champion_embedding = Embedding(
            num_indices=161, embedding_dims=256, num_ones=1
        )
        self.summoner_spell_embedding = Embedding(
            num_indices=16, embedding_dims=32, num_ones=2
        )
        self.rune_embedding = Embedding(num_indices=63, embedding_dims=64, num_ones=6)
        self.queue_embedding = Embedding(num_indices=51, embedding_dims=64, num_ones=1)
        self.player_embedding = nn.Sequential(
            BNLinear(352, 4096, 0.1),
            BNLinear(4096, 4096, 0.1),
            BNLinear(4096, 4096, 0.1),
        )
        self.game_predictor = nn.Sequential(
            BNLinear(2 * 4096 + 64, 8192, 0.1),
            BNLinear(8192, 8192, 0.1),
            BNLinear(8192, 8192, 0.1),
            nn.Linear(8192, 1, bias=False),
            nn.Sigmoid(),
        )

    def embed_player(self, player: dict) -> torch.Tensor:
        player = torch.concat(
            [
                self.champion_embedding(player["champion"]),
                self.summoner_spell_embedding(player["summonerSpells"]),
                self.rune_embedding(player["runes"]),
            ],
            dim=-1,
        )
        return self.player_embedding(player)

    def embed_team(self, team: list[torch.Tensor]) -> torch.Tensor:
        player_embeddings = [self.embed_player(player) for player in team]
        team_embedding = torch.stack(player_embeddings, dim=2)
        team_embedding = torch.mean(team_embedding, dim=2)
        return team_embedding

    def embed_game(self, game) -> torch.Tensor:
        team1_embeddings = self.embed_team(game["team1"])
        team2_embeddings = self.embed_team(game["team2"])
        return torch.concat(
            [team1_embeddings, team2_embeddings, self.queue_embedding(game["queue"])],
            dim=-1,
        )

    def forward(self, game) -> torch.Tensor:
        game_embeddings = self.embed_game(game)
        return self.game_predictor(game_embeddings)
