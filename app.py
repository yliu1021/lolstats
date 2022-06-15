import os

import torch

from flask import Flask, request
from riotapi import LolClient, HTTPError
from dotenv import load_dotenv

from train import models, transforms

# Load environment file
load_dotenv()
riot_key = os.getenv("RIOT_KEY")

app = Flask(__name__)

model = models.MatchModel()
model.load_state_dict(torch.load("./models/model_1.pt"))
model.eval()


def predict_game(game):
    if game["gameQueueConfigId"] not in [400, 420, 430, 440, 700]:
        return {"error": "unsupported game mode"}
    participants = {100: [], 200: []}
    for participant in game["participants"]:
        participants[participant["teamId"]].append(
            {
                "championId": participant["championId"],
                "summonerSpellIds": [participant["spell1Id"], participant["spell2Id"]],
                "runeIds": [
                    perk_id
                    for perk_id in participant["perks"]["perkIds"]
                    if perk_id > 5100
                ],
                "summonerName": participant["summonerName"]
            }
        )
    if len(participants) != 2:
        return {"error": "Expected participants to come from two teams"}
    X = {
        "game": {"team1": participants[100], "team2": participants[200]},
        "team1Won": 0
    }
    X = transforms.ToTensor()({
        "game": {"team1": participants[100], "team2": participants[200]},
        "team1Won": 0
    })["game"]
    y = model(X)
    return {
        "team1": participants[100],
        "team2": participants[200],
        "prediction": y.item()
    }


@app.get("/live")
async def get_live_game():
    summoner_name = request.args.get("summoner", type=str)
    if summoner_name is None:
        return {"error": "missing summoner query parameter"}, 400
    async with LolClient(riot_key) as lc:
        try:
            summoner = await lc.summoner.by_name(summoner_name)
        except HTTPError:
            return {"error": "summoner not found"}, 404
        try:
            game = await lc.spectator.by_summoner(summoner["id"])
        except HTTPError as e:
            print(e)
            return {"error": "game not found"}, 404
        return predict_game(game)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
