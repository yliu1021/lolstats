import os
from pprint import pprint

import requests
import riotwatcher
from dotenv import load_dotenv
from flask import Flask, request

# Load environment file
load_dotenv()
riot_key = os.getenv("RIOT_KEY")

# Load riot watcher
lw = riotwatcher.LolWatcher(riot_key)
region = "NA1"  # set to NA1 by default

# Load definitions
queue_ids = {}
for queue_def in requests.get(
    "https://static.developer.riotgames.com/docs/lol/queues.json"
).json():
    queue_ids[queue_def["queueId"]] = queue_def


# Setup flask
app = Flask(__name__)


@app.route("/match_history", methods=["GET"])
def match_history():
    args = request.args.to_dict()
    summoner = args.get("summoner")
    if summoner is None:
        return {"error": "summoner name missing"}, 400
    try:
        summoner = lw.summoner.by_name(region=region, summoner_name=summoner)
    except riotwatcher.ApiError:
        return {"error": "failed to fetch summoner"}, 404
    matches = lw.match.matchlist_by_puuid(region=region, puuid=summoner["puuid"])
    return {"matches": matches}


@app.route("/match", methods=["GET"])
def match():
    args = request.args.to_dict()
    match_id = args.get("id")
    if match_id is None:
        return {"error", "match id missing"}, 400
    try:
        match = lw.match.by_id(region=region, match_id=match_id)
    except riotwatcher.ApiError:
        return {"error", "failed to fetch match"}, 404
    match_info = match["info"]
    queue_id = match_info["queueId"]
    teams = match_info["teams"]
    participants = [
        {
            "teamId": x["teamId"],
            "summonerName": x["summonerName"],
            "championName": x["championName"],
            "kills": x["kills"],
            "deaths": x["deaths"],
            "assists": x["assists"],
        }
        for x in match_info["participants"]
    ]
    return {
        "gameCreation": match_info["gameCreation"],
        "gameDuration": match_info["gameDuration"],
        "map": queue_ids[queue_id]["map"],
        "queue": queue_ids[queue_id]["description"],
        "teams": teams,
        "participants": participants,
    }
