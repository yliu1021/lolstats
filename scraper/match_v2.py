"""
Scrapes matches and timelines
"""
import asyncio
from datetime import datetime
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

seen_summoners = set()
summoners_to_scrape = asyncio.Queue()
matches_to_scrape = asyncio.Queue()
num_matches_scraped = 0
num_summoners_scraped = 0


last_rate_limit = datetime.now()
async def rate_limit():
    sec_since_last_rate_limit = (datetime.now() - last_rate_limit).total_seconds()
    if sec_since_last_rate_limit < 1:
        await asyncio.sleep(random.random() * 10) + 1
    elif sec_since_last_rate_limit < 5:
        await asyncio.sleep(random.random() * 2
        ) + 1
    last_rate_limit = datetime.now()


async def report_progress():
    while True:
        now = datetime.now().strftime("%H:%M:%S")
        print(
            f"\r[{now}] {num_matches_scraped} matches scraped | {num_summoners_scraped} summoners "
            f"{summoners_to_scrape.qsize()=} {matches_to_scrape.qsize()=}",
            end=""
        )
        await asyncio.sleep(1)


async def scrape_match(lc: LolClient):
    global num_matches_scraped
    while True:
        match_id = await matches_to_scrape.get()
        match = await matches_collection.find_one({"_id": match_id})
        if match is not None:
            participants = match["metadata"]["participants"]
        else:
            try:
                match = await lc.match.match(match_id=match_id)
                match["_id"] = match["metadata"]["matchId"]
                try:
                    await matches_collection.insert_one(match)
                except errors.DuplicateKeyError:
                    pass
                num_matches_scraped += 1
                participants = match["metadata"]["participants"]
            except HTTPError as e:
                if e.status == 429:
                    await matches_to_scrape.put(match_id)
                    await rate_limit()
                participants = []
        if summoners_to_scrape.qsize() > 10_000:
            continue
        random.shuffle(participants)
        for puuid in participants:
            if puuid in seen_summoners:
                continue
            seen_summoners.add(puuid)
            await summoners_to_scrape.put(puuid)


async def scrape_summoner(lc: LolClient):
    global num_summoners_scraped
    while True:
        summoner_puuid = await summoners_to_scrape.get()
        try:
            matchlist = await lc.match.match_history(
                puuid=summoner_puuid, start_time=1655989200, count=100
            )
            if matches_to_scrape.qsize() > 10_000:
                continue
            random.shuffle(matchlist)
            for match_id in matchlist:
                await matches_to_scrape.put(match_id)
            num_summoners_scraped += 1
        except HTTPError as e:
            if e.status == 429:
                await summoners_to_scrape.put(summoner_puuid)
                await rate_limit()


async def main():
    async with LolClient(riot_key) as lc:
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
            await summoners_to_scrape.put(summ["puuid"])

        match_scrapers = {asyncio.create_task(scrape_match(lc)) for _ in range(4)}
        summoner_scrapers = {asyncio.create_task(scrape_summoner(lc)) for _ in range(4)}
        report_progress_task = asyncio.create_task(report_progress())

        await asyncio.gather(
            asyncio.wait(match_scrapers, return_when=asyncio.FIRST_EXCEPTION),
            asyncio.wait(summoner_scrapers, return_when=asyncio.FIRST_EXCEPTION),
            report_progress_task
        )


if __name__ == "__main__":
    asyncio.run(main())
