import time
import pathlib
import pickle
import functools

from pymongo.mongo_client import MongoClient
from torch.utils.data import Dataset


def _process_match(match):
    match_info = match["info"]
    teams = {}
    for team in match_info["teams"]:
        teams[team["teamId"]] = team
    assert len(teams) == 2, f"Expected two teams, got:\n{teams}\n\n{match}"
    assert 100 in teams and 200 in teams, "Expected team ids to be 100 and 200"
    participants = {100: [], 200: []}
    for participant in match_info["participants"]:
        participants[participant["teamId"]].append(
            {
                "championId": participant["championId"],
                "summonerSpellIds": [
                    participant["summoner1Id"],
                    participant["summoner2Id"],
                ],
                "runeIds": [
                    selection["perk"]
                    for style in participant["perks"]["styles"]
                    for selection in style["selections"]
                ],
            }
        )
    assert len(participants) == 2, "Expected participants to come from two teams"
    assert len(participants[200]) == len(participants[100]) == 5, "Expected 5 players on each team"
    return {
        "game": {
            "team1": participants[100],
            "team2": participants[200],
            "queue": match_info["queueId"],
        },
        "team1Won": int(teams[100]["win"]),
    }


class MatchesDataset(Dataset):
    def __init__(
        self, data_loc: str | pathlib.Path, game_transforms=None, sample_transforms=None
    ) -> None:
        super().__init__()
        self.game_transforms = [] if game_transforms is None else game_transforms
        self.sample_transforms = [] if sample_transforms is None else sample_transforms
        data_loc = pathlib.Path(data_loc)
        data_loc.mkdir(exist_ok=True)
        data_path = data_loc / "matches_v1.pickle"
        if data_path.exists():
            with open(data_path, "rb") as data_file:
                self.matches = pickle.load(data_file)
        else:
            client = MongoClient("localhost", 27017)
            db = client["lolstats"]
            collection = db["matches"]
            matches = collection.find(
                {
                    # "info.participants.0.gameEndedInSurrender": False,
                    "info.participants.0.gameEndedInEarlySurrender": False,
                    "info.queueId": {"$lt": 2000, "$gt": 320},
                    "info.gameVersion": {"$regex": "^12\.14.*"},
                }
            )
            print("Loading matches...")
            self.matches = []
            for match in matches:
                self.matches.append(_process_match(match))
                print(f"\rLoaded {len(self.matches)}", end="")
            print()
            print("Saving...")
            with open(data_path, "wb") as data_file:
                pickle.dump(self.matches, data_file)
            print("Done")

    def __len__(self) -> int:
        return len(self.matches)

    @functools.cache
    def __getitem__(self, index):
        sample = self.matches[index]
        for transform in self.game_transforms:
            sample["game"] = transform(sample["game"])
        for transform in self.sample_transforms:
            sample = transform(sample)
        return sample


if __name__ == "__main__":
    start_time = time.time()
    dataset = MatchesDataset("./data")
    end_time = time.time()
    from pprint import pprint

    print(f"Loaded {len(dataset)} matches in {end_time - start_time:.2f} seconds")
    pprint(dataset[0])
