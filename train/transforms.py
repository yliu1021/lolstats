import json
import random

import torch


class ToTensor:
    """Converts game data to tensor"""

    def __init__(self) -> None:
        champions = json.load(open("./datadragon/champion.json", "r"))
        champions = champions["data"]
        champions = dict(
            [(int(champion["key"]), champion) for champion in champions.values()]
        )
        self.champion_ids = list(champions.keys())
        self.num_champions = len(self.champion_ids)

    def _encode_champion(self, champion_id: int) -> torch.Tensor:
        ind = self.champion_ids.index(champion_id)
        encoding = torch.zeros(self.num_champions, dtype=torch.float32)
        encoding[ind] = 1
        return encoding

    def _encode_team(self, team: list[dict]) -> torch.Tensor:
        return torch.concat(
            [self._encode_champion(player["championId"]) for player in team]
        )

    def __call__(self, sample):
        game = torch.concat(
            [
                self._encode_team(sample["game"]["team1"]),
                self._encode_team(sample["game"]["team2"]),
            ]
        )
        return {"game": game, "team1Won": torch.tensor([sample["team1Won"]], dtype=torch.float32)}


class TeamShuffle:
    def __call__(self, sample):
        if random.random() < 0.5:
            random.shuffle(sample["game"]["team1"])
            random.shuffle(sample["game"]["team2"])
        return sample
