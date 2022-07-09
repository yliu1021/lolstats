import asyncio
from dataclasses import dataclass
import os
import sys
import json

from riotapi import LolClient
from dotenv import load_dotenv
import torch

import train


# Load environment file
load_dotenv()
riot_key = os.getenv("RIOT_KEY")

champions_ = json.load(open("./datadragon/12.12.1/champion.json", "r", encoding="utf8"))
champions_ = champions_["data"]
champions_ = dict((int(champion["key"]), champion) for champion in champions_.values())


@dataclass
class Player:
    name: str  # summoner name
    champion: str  # champion name

    participant_info: dict

    def __init__(self, participant_info) -> None:
        self.participant_info = participant_info
        self.name = participant_info["summonerName"]
        self.champion_id = participant_info["championId"]
        champion = champions_[self.champion_id]
        self.champion = champion["name"]

    def to_json(self) -> dict:
        return {
            **self.participant_info,
            "champion": self.champion
        }
    
    def __repr__(self) -> str:
        return f"{self.name} ({self.champion})"


@dataclass
class Team:
    players: list[Player]

    def __init__(self, participants):
        self.players = [Player(participant) for participant in participants]

    def __contains__(self, summoner_name: str) -> bool:
        for player in self.players:
            if player.name == summoner_name:
                return True
        return False

    def to_json(self) -> list[dict]:
        return [player.to_json() for player in self.players]


@dataclass
class Game:
    friendly_team: Team
    enemy_team: Team
    queue_id: int

    def __init__(self, game_info, summoner_name):
        participants = {100: [], 200: []}
        for participant in game_info["participants"]:
            participants[participant["teamId"]].append(
                {
                    "championId": participant["championId"],
                    "summonerSpellIds": [
                        participant["spell1Id"],
                        participant["spell2Id"],
                    ],
                    "runeIds": [
                        perk_id
                        for perk_id in participant["perks"]["perkIds"]
                        if perk_id > 5100
                    ],
                    "summonerName": participant["summonerName"],
                }
            )
        assert len(participants) == 2, "Expected participants to come from two teams"
        team_1 = Team(participants[100])
        team_2 = Team(participants[200])
        if summoner_name in team_1:
            self.friendly_team = team_1
            self.enemy_team = team_2
        else:
            assert summoner_name in team_2, "Player not found on either teams"
            self.friendly_team = team_2
            self.enemy_team = team_1
        self.queue_id = game_info["gameQueueConfigId"]

    def to_json(self) -> dict:
        return {
            "team1": self.friendly_team.to_json(),
            "team2": self.enemy_team.to_json(),
            "queue": self.queue_id,
        }


class Predictor:
    def __init__(self, model_path: str) -> None:
        self.model = train.models.MatchModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def make_batch(self, d):
        if isinstance(d, torch.Tensor):
            return d[None, :]
        elif isinstance(d, list):
            return [self.make_batch(x) for x in d]
        elif isinstance(d, dict):
            return {k: self.make_batch(v) for k, v in d.items()}
        else:
            raise ValueError(f"Unsupported type: {type(d)}")

    async def predict(self, summoner_name: str) -> tuple[Game, float]:
        async with LolClient(riot_key) as lc:
            summoner = await lc.summoner.by_name(summoner_name)
            game_info = await lc.spectator.by_summoner(summoner["id"])
        game = Game(game_info, summoner_name)
        game_tensor = train.transforms.GameToTensor()(game.to_json())
        game_tensor = self.make_batch(game_tensor)
        pred = self.model(game_tensor)
        pred = pred.item()
        return game, pred


async def main(summoner_name: str):
    predictor = Predictor("models/20220628_03-29-50/epoch_9/model.pt")
    game, pred = await predictor.predict(summoner_name)
    print(game)
    print(f"Chances of winning: {pred}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
