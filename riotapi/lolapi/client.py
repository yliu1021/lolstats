import aiohttp


class ClientAPI:
    def __init__(self):
        pass

    async def all_game_data(self) -> dict:
        url = "https://127.0.0.1:2999/liveclientdata/allgamedata"
        async with aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(verify_ssl=False)
        ) as session:
            async with session.get(url) as response:
                return await response.json()
