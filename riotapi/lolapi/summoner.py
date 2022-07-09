from urllib.parse import quote
from riotapi import api


class SummonerAPI:
    def __init__(self, base: api.BaseAPI):
        self._base = base

    async def by_name(self, summoner_name: str) -> dict:
        summoner_name = quote(summoner_name)
        url = f"https://na1.api.riotgames.com/lol/summoner/v4/summoners/by-name/{summoner_name}"
        return await self._base.get(url)
