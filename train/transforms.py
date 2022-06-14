import json
import random

import torch


class ToTensor:
    """Converts game data to tensor"""

    def __init__(self) -> None:
        champions = json.load(open("./datadragon/champion.json", "r"))
        champions = champions["data"]
        champions = dict(
            (int(champion["key"]), champion) for champion in champions.values()
        )
        self.champion_ids = sorted(list(champions.keys()))
        self.num_champions = len(self.champion_ids)

        summoner_spells = json.load(open("./datadragon/summoner.json", "r"))
        summoner_spells = summoner_spells["data"]
        summoner_spells = dict(
            (int(summoner_spell["key"]), summoner_spell)
            for summoner_spell in summoner_spells.values()
        )
        self.summoner_spell_ids = sorted(list(summoner_spells.keys()))
        self.num_summoner_spells = len(self.summoner_spell_ids)

    def _encode_player(self, player: dict) -> torch.Tensor:
        champion_id = player["championId"]
        ind = self.champion_ids.index(champion_id)
        champion_encoding = torch.zeros(self.num_champions, dtype=torch.float32)
        champion_encoding[ind] = 1
        summoner_spell_id1 = self.summoner_spell_ids.index(player["summoner1Id"])
        summoner_spell_id2 = self.summoner_spell_ids.index(player["summoner2Id"])
        summoner_spell_encoding = torch.zeros(self.num_summoner_spells, dtype=torch.float32)
        summoner_spell_encoding[summoner_spell_id1] = 1
        summoner_spell_encoding[summoner_spell_id2] = 1
        return torch.concat([champion_encoding, summoner_spell_encoding])

    def _encode_team(self, team: list[dict]) -> torch.Tensor:
        return [self._encode_player(player) for player in team]

    def __call__(self, sample):
        game = sample["game"]
        game = {
            "team1": self._encode_team(game["team1"]),
            "team2": self._encode_team(game["team2"]),
        }
        return {
            "game": game,
            "team1Won": torch.tensor([sample["team1Won"]], dtype=torch.float32),
        }


class TeamShuffle:
    def __call__(self, sample):
        if random.random() < 0.5:
            random.shuffle(sample["game"]["team1"])
            random.shuffle(sample["game"]["team2"])
        return sample
