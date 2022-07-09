from flask import Flask, request, send_file

from predict import Predictor


predictor = Predictor("models/model.pt")

app = Flask(__name__, static_folder="static", static_url_path="")


@app.route("/", methods=["GET"])
def index():
    return send_file("static/index.html")


@app.route("/game.js", methods=["GET"])
def game_js():
    return send_file("static/game.js")


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
        return {"error": "Match not found."}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
