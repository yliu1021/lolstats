from typing import Optional, Dict
from riotapi import api


class MatchAPI:
    def __init__(self, base: api.BaseAPI):
        self._base = base

    async def match_history(
        self,
        puuid: str,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        queue: Optional[int] = None,
        type: Optional[str] = None,
        start: Optional[int] = None,
        count: Optional[int] = None,
    ) -> Dict:
        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/by-puuid/{puuid}/ids"
        params = {}
        if start_time is not None:
            params["startTime"] = str(start_time)
        if end_time is not None:
            params["endTime"] = str(end_time)
        if queue is not None:
            params["queue"] = str(queue)
        if type is not None:
            params["type"] = type
        if start is not None:
            params["start"] = str(start)
        if count is not None:
            params["count"] = str(count)
        if not params:
            params = None
        return await self._base.get(url, params=params)

    async def match(self, match_id: str) -> Dict:
        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}"
        return await self._base.get(url)

    async def timeline(self, match_id: str) -> Dict:
        url = f"https://americas.api.riotgames.com/lol/match/v5/matches/{match_id}/timeline"
        return await self._base.get(url)
