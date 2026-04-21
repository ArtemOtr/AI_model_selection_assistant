import asyncio
from typing import Any, Dict
from agents.mws_scraper import MWSTableScraper
import pandas as pd


'''

input agent
асинхронный класс, который будет чекать умер кэш или нет и по таймеру его внутри себя
но основной функцией будет выдавать json с модельками
'''

class ParserCacheAgent():
    def __init__(self, ttl: int = 300, models_info_url = "https://mws.ru/docs/cloud-platform/gpt/general/gpt-models.html", models_cost_url = "https://mws.ru/docs/cloud-platform/gpt/general/pricing.html" ):
        self._ttl = ttl
        self._scraper = MWSTableScraper(models_info_url, models_cost_url)
        self._cache_models_table: pd.DataFrame = self._scraper.get_merged_tables()
        self._cache_models_dict: Dict[str, Any] = self._cache_models_table.to_dict(orient="records")
        self._lock = asyncio.Lock()
        self._running = False
        self._watchdog_task: asyncio.Task | None = None

        try:
            asyncio.get_running_loop()
            asyncio.create_task(self.start())
        except RuntimeError:
            pass

    async def start(self):
        if self._running:
            return
        self._running = True
        self._watchdog_task = asyncio.create_task(self._watchdog())

    async def stop(self):
        self._running = False
        if self._watchdog_task is not None:
            self._watchdog_task.cancel()
            try:
                await self._watchdog_task
            except asyncio.CancelledError:
                pass
            self._watchdog_task = None

    async def _watchdog(self):
        while self._running:
            await asyncio.sleep(self._ttl)
            try:
                table = await asyncio.to_thread(self._scraper.get_merged_tables)
            except Exception:

                continue

            async with self._lock:
                self._cache_models_table = table
                self._cache_models_dict = table.to_dict(orient="records")

    def get_models_table(self):
        return self._cache_models_table

    def get_models_dict(self):
        return self._cache_models_dict
