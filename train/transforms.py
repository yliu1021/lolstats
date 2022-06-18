import json
import random

import torch


class GameToTensor:
    """Converts game data to tensor"""

    def __init__(self) -> None:
        champions = json.load(open("./datadragon/12.11.1/champion.json", "r"))
        champions = champions["data"]
        champions = dict(
            (int(champion["key"]), champion) for champion in champions.values()
        )
        self.champion_ids = dict(
            (id, ind) for ind, id in enumerate(sorted(champions.keys()))
        )
        self.num_champions = len(self.champion_ids)

        summoner_spells = json.load(open("./datadragon/12.11.1/summoner.json", "r"))
        summoner_spells = summoner_spells["data"]
        summoner_spells = dict(
            (int(summoner_spell["key"]), summoner_spell)
            for summoner_spell in summoner_spells.values()
        )
        self.summoner_spell_ids = dict(
            (id, ind) for ind, id in enumerate(sorted(summoner_spells.keys()))
        )
        self.num_summoner_spells = len(self.summoner_spell_ids)

        runes_raw = json.load(open("./datadragon/12.11.1/runesReforged.json", "r"))
        runes = {}
        for rune_page in runes_raw:
            for rune_slots in rune_page["slots"]:
                for rune in rune_slots["runes"]:
                    runes[rune["id"]] = rune
        self.rune_ids = dict((id, ind) for ind, id in enumerate(sorted(runes.keys())))
        self.num_runes = len(self.rune_ids)

        queue_ids = json.load(open("./datadragon/queues.json", "r"))
        queue_ids = dict(
            (queue["queueId"], queue)
            for queue in queue_ids
            if (queue["notes"] is None or "deprecated" not in queue["notes"].lower())
        )
        self.queue_ids = dict(
            (id, ind) for ind, id in enumerate(sorted(queue_ids.keys()))
        )
        self.num_queues = len(self.queue_ids)

    def _encode_player(self, player: dict) -> torch.Tensor:
        champion_encoding = torch.zeros(self.num_champions, dtype=torch.float32)
        champion_encoding[self.champion_ids[player["championId"]]] = 1

        summoner_spell_encoding = torch.zeros(
            self.num_summoner_spells, dtype=torch.float32
        )
        for summ_id in player["summonerSpellIds"]:
            if summ_id in self.summoner_spell_ids:
                summoner_spell_encoding[self.summoner_spell_ids[summ_id]] = 1

        rune_encoding = torch.zeros(self.num_runes, dtype=torch.float32)
        for rune_id in player["runeIds"]:
            rune_encoding[self.rune_ids[rune_id]] = 1

        summoner_level = torch.tensor(
            [player["summonerLevel"] / 500], dtype=torch.float32
        )  # normalize by 500

        return torch.concat(
            [champion_encoding, summoner_spell_encoding, rune_encoding, summoner_level]
        )

    def _encode_team(self, team: list[dict]) -> torch.Tensor:
        return [self._encode_player(player) for player in team]

    def _encode_queue_id(self, queue_id: int) -> torch.Tensor:
        queue_encoding = torch.zeros(self.num_queues, dtype=torch.float32)
        queue_encoding[self.queue_ids[queue_id]] = 1
        return queue_encoding

    def __call__(self, game):
        game = {
            "team1": self._encode_team(game["team1"]),
            "team2": self._encode_team(game["team2"]),
            "queue": self._encode_queue_id(game["queue"]),
        }
        return game


class ToTensor:
    def __init__(self) -> None:
        self.game_transform = GameToTensor()

    def __call__(self, sample):
        return {
            "game": self.game_transform(sample["game"]),
            "team1Won": torch.tensor([sample["team1Won"]], dtype=torch.float32),
        }


class TeamShuffle:
    def __call__(self, sample):
        if random.random() < 0.5:
            return {
                "game": {
                    "team1": sample["game"]["team2"],
                    "team2": sample["game"]["team1"],
                    "queue": sample["game"]["queue"],
                },
                "team1Won": 1 - sample["team1Won"],
            }
        else:
            return sample
