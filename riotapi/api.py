from typing import Optional, Dict
import aiohttp


class HTTPError(Exception):
    def __init__(self, status: int, reason: str) -> None:
        self.status = status
        self.reason = reason


class BaseAPI:
    """BaseAPI implements basic API functionality"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session = aiohttp.ClientSession(headers={"X-Riot-Token": api_key})

    async def __aenter__(self) -> "BaseAPI":
        await self._session.__aenter__()
        return self

    async def __aexit__(self, *args, **kwargs):
        await self._session.__aexit__(*args, **kwargs)

    async def get(self, url: str, params: Optional[Dict[str, str]] = None) -> Dict:
        async with self._session.get(url, params=params) as response:
            if not response.ok:
                raise HTTPError(response.status, response.reason)
            return await response.json()
