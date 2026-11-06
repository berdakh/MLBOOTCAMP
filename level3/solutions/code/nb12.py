"""Reference solutions for Level 3 notebook 12. Graded by tests/test_solutions.py."""

from __future__ import annotations

import asyncio
import time


class ex1:  # noqa: N801 - graded as ``ex1``; in the notebook it is called DynamicBatcher
    """Collect requests into batches of at most ``max_batch_size``, waiting at most ``max_wait_ms``."""

    def __init__(self, process_batch, max_batch_size=16, max_wait_ms=5.0):
        self.process_batch = process_batch
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = None

    def _q(self):
        if self.queue is None:  # create it inside the running event loop
            self.queue = asyncio.Queue()
        return self.queue

    async def submit(self, item):
        fut = asyncio.get_running_loop().create_future()
        await self._q().put((item, fut))
        return await fut

    async def run(self):
        q = self._q()
        while True:
            batch = [await q.get()]  # wait (as long as needed) for the first request
            deadline = time.perf_counter() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(q.get(), remaining))
                except asyncio.TimeoutError:
                    break
            results = self.process_batch([item for item, _ in batch])  # one model call per batch
            for (_, fut), res in zip(batch, results):
                if not fut.done():
                    fut.set_result(res)


def ex2(requests_per_second, price_per_hour, utilization):
    return 1000 * price_per_hour / (requests_per_second * 3600 * utilization)
