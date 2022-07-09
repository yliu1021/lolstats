from typing import Dict
from riotapi import api


class SpectatorAPI:
    def __init__(self, base: api.BaseAPI):
        self._base = base

    async def by_summoner(self, encrypted_summoner_id: str) -> Dict:
        url = f"https://na1.api.riotgames.com/lol/spectator/v4/active-games/by-summoner/{encrypted_summoner_id}"
        return await self._base.get(url)
