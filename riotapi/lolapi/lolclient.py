from riotapi import api
from .summoner import SummonerAPI
from .match import MatchAPI
from .spectator import SpectatorAPI
from .client import ClientAPI


class LolClient:
    def __init__(self, api_key: str):
        self._base = api.BaseAPI(api_key)
        self.summoner = SummonerAPI(self._base)
        self.match = MatchAPI(self._base)
        self.spectator = SpectatorAPI(self._base)
        self.client = ClientAPI()

    async def __aenter__(self) -> "LolClient":
        await self._base.__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self._base.__aexit__(*args, **kwargs)
