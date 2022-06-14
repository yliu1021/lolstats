import json

import numpy as np


champions = json.load(open("./datadragon/champion.json", "r"))
champions = champions["data"]
champions = dict([(int(champion["key"]), champion) for champion in champions.values()])
champion_ids = list(champions.keys())
num_champions = len(champion_ids)


def encode(champion_id: int) -> np.ndarray:
    ind = champion_ids.index(champion_id)
    encoding = np.zeros(num_champions, dtype=np.float32)
    encoding[ind] = 1
    return encoding


if __name__ == "__main__":
    from pprint import pprint
    pprint(champions)
