import json
import random

import torch


class ToTensor:
    """Converts game data to tensor"""

    def __init__(self) -> None:
        champions = json.load(open("./datadragon/12.11.1/champion.json", "r"))
        champions = champions["data"]
        champions = dict(
            (int(champion["key"]), champion) for champion in champions.values()
        )
        self.champion_ids = sorted(list(champions.keys()))
        self.num_champions = len(self.champion_ids)

        summoner_spells = json.load(open("./datadragon/12.11.1/summoner.json", "r"))
        summoner_spells = summoner_spells["data"]
        summoner_spells = dict(
            (int(summoner_spell["key"]), summoner_spell)
            for summoner_spell in summoner_spells.values()
        )
        self.summoner_spell_ids = sorted(list(summoner_spells.keys()))
        self.num_summoner_spells = len(self.summoner_spell_ids)

        runes_raw = json.load(open("./datadragon/12.11.1/runesReforged.json", "r"))
        runes = {}
        for rune_page in runes_raw:
            for rune_slots in rune_page["slots"]:
                for rune in rune_slots["runes"]:
                    runes[rune["id"]] = rune
        self.rune_ids = sorted(list(runes.keys()))
        self.num_runes = len(self.rune_ids)

    def _encode_player(self, player: dict) -> torch.Tensor:
        champion_encoding = torch.zeros(self.num_champions, dtype=torch.float32)
        champion_encoding[self.champion_ids.index(player["championId"])] = 1

        summoner_spell_encoding = torch.zeros(
            self.num_summoner_spells, dtype=torch.float32
        )
        for summ_id in player["summonerSpellIds"]:
            summoner_spell_encoding[self.summoner_spell_ids.index(summ_id)] = 1

        rune_encoding = torch.zeros(self.num_runes, dtype=torch.float32)
        for rune_id in player["runeIds"]:
            rune_encoding[self.rune_ids.index(rune_id)] = 1

        return torch.concat([champion_encoding, summoner_spell_encoding, rune_encoding])

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


if __name__ == "__main__":
    x = ToTensor()
