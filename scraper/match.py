"""
Scrapes a bunch of matches and stores them
"""
import asyncio
import os
import random

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import errors
from riotapi import LolClient, HTTPError
from dotenv import load_dotenv

# Load environment file
load_dotenv()
riot_key = os.getenv("RIOT_KEY")

# Load mongodb client
client = AsyncIOMotorClient(host="localhost", port=27017)
db = client["lolstats"]
matches_collection = db["matches"]


async def main():
    async with LolClient(riot_key) as lc:
        puuids_to_scrape = set()
        for summoner_names in [
            "trololololol123",
            "NinjaBIade",
            "Toujours Seule",
            "dkim1206",
            "slimebloobblob",
            "sarmeges",
            "Nunu x Willump",
            "BSIZZLEMONEY",
        ]:
            summ = await lc.summoner.by_name(summoner_name=summoner_names)
            puuids_to_scrape.add(summ["puuid"])
        seen_matches = set()
        seen_puuids = set()

        async def scrape_match(match_id: str) -> list[str]:
            match = await matches_collection.find_one({"_id": match_id})
            if match is not None:
                return match["metadata"]["participants"]
            match = await lc.match.match(match_id=match_id)
            match["_id"] = match["metadata"]["matchId"]
            try:
                await matches_collection.insert_one(match)
            except errors.DuplicateKeyError:
                pass
            return match["metadata"]["participants"]

        while True:
            puuid = puuids_to_scrape.pop()
            if puuid in seen_puuids:
                continue
            seen_puuids.add(puuid)
            matchlist = await lc.match.match_history(
                puuid=puuid, start_time=1655989200, count=100
            )
            matchlist = [m for m in matchlist if m not in seen_matches]
            random.shuffle(matchlist)
            should_retry = True
            while should_retry:
                should_retry = False
                try:
                    all_puuids = await asyncio.gather(
                        *[
                            asyncio.Task(scrape_match(match_id))
                            for match_id in matchlist
                        ]
                    )
                    for puuids in all_puuids:
                        puuids_to_scrape = puuids_to_scrape.union(puuids)
                    for match_id in matchlist:
                        seen_matches.add(match_id)
                    print(
                        f"\rScraped {len(seen_matches)} matches, across {len(seen_puuids)} summoners",
                        end="",
                        flush=True,
                    )
                    # await asyncio.sleep(2*random.random() + 1)
                except HTTPError as e:
                    if e.status == 429:
                        print("Rate limited")
                        await asyncio.sleep(5)
                        should_retry = True


if __name__ == "__main__":
    asyncio.run(main())
