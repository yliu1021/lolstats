import time
import pathlib
import pickle

from pymongo.mongo_client import MongoClient
from torch.utils.data import Dataset


def _process_match(match):
    match_info = match["info"]
    teams = {}
    for team in match_info["teams"]:
        teams[team["teamId"]] = team
    assert len(teams) == 2, f"Expected two teams, got {teams} {match}"
    assert 100 in teams and 200 in teams, "Expected team ids to be 100 and 200"
    participants = {100: [], 200: []}
    for participant in match_info["participants"]:
        participants[participant["teamId"]].append(
            {"championId": participant["championId"]}
        )
    assert len(participants) == 2, "Expected participants to come from two teams"
    return {
        "game": {"team1": participants[100], "team2": participants[200]},
        "team1Won": int(teams[100]["win"]),
    }


class MatchesDataset(Dataset):
    def __init__(self, data_loc: str | pathlib.Path, transforms=None) -> None:
        super().__init__()
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = []
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
                    "info.queueId": {"$in": [400, 420, 430, 440, 700]},
                    "info.gameVersion": {"$regex": "^12\.11.*"},
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

    def __getitem__(self, index):
        match = self.matches[index]
        for transform in self.transforms:
            match = transform(match)
        return match


if __name__ == "__main__":
    start_time = time.time()
    dataset = MatchesDataset("./data")
    end_time = time.time()
    from pprint import pprint
    print(f"Loaded {len(dataset)} matches in {end_time - start_time:.2f} seconds")
    pprint(dataset[0])