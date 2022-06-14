import os

from flask import Flask, request
from riotapi import LolClient, HTTPError
from dotenv import load_dotenv

# Load environment file
load_dotenv()
riot_key = os.getenv("RIOT_KEY")

app = Flask(__name__)


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
        return summoner
