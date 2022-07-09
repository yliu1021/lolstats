import os

from flask import Flask, request
from dotenv import load_dotenv

from predict import Predictor

# Load environment file
load_dotenv()
riot_key = os.getenv("RIOT_KEY")

app = Flask(__name__)

predictor = Predictor("models/20220708_23-12-18/epoch_46/model.pt")


@app.get("/live")
async def get_live_game():
    summoner_name = request.args.get("summoner", type=str)
    if summoner_name is None:
        return {"error": "missing summoner query parameter"}, 400
    try:
        game, pred = await predictor.predict(summoner_name)
        return {
            "game": game.to_json(),
            "chancesOfWinning": pred
        }
    except:
        return {}, 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
