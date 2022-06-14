"""
Scrapes a bunch of matches and stores them
"""
import asyncio
import sys
import os

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


async def main(first_summoner: str):
    async with LolClient(riot_key) as lc:
        root_summ = await lc.summoner.by_name(summoner_name=first_summoner)
        puuids_to_scrape = {root_summ["puuid"]}
        seen_matches = set()
        seen_puuids = set()

        async def scrape_match(match_id: str) -> list[str]:
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
            matchlist = await lc.match.match_history(puuid=puuid, count=20)
            matchlist = [m for m in matchlist if m not in seen_matches]
            should_retry = True
            while should_retry:
                should_retry = False
                try:
                    all_puuids = await asyncio.gather(*[asyncio.Task(scrape_match(match_id)) for match_id in matchlist])
                    for puuids in all_puuids:
                        puuids_to_scrape = puuids_to_scrape.union(puuids)
                    for match_id in matchlist:
                        seen_matches.add(match_id)
                    print(f"\rScraped {len(seen_matches)} matches, across {len(seen_puuids)} summoners", end="", flush=True)
                except HTTPError as e:
                    if e.status == 429:
                        await asyncio.sleep(5)
                        should_retry = True


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
