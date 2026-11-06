"""Self-checks for Level 3 notebook 12 · High-throughput serving."""

from __future__ import annotations

from churnlab.checks import close, run_check


def ex1(DynamicBatcher):
    def test():
        import asyncio
        import time

        from scaleml.serving import run_async

        calls: list[list] = []

        def process(items):
            calls.append(list(items))
            return [x * 10 for x in items]

        async def scenario():
            b = DynamicBatcher(process, max_batch_size=4, max_wait_ms=60)
            task = asyncio.create_task(b.run())
            try:
                burst = await asyncio.wait_for(asyncio.gather(*(b.submit(i) for i in range(10))), 5)
                n_burst = len(calls)
                t0 = time.perf_counter()
                lone = await asyncio.wait_for(b.submit(99), 5)
                lone_s = time.perf_counter() - t0

                async def later(x, delay):
                    await asyncio.sleep(delay)
                    return await b.submit(x)

                n_before = len(calls)
                staggered = await asyncio.wait_for(
                    asyncio.gather(later(100, 0.0), later(101, 0.01), later(102, 0.02)), 5)
                return burst, calls[:n_burst], lone, lone_s, staggered, calls[n_before:]
            finally:
                task.cancel()
                try:
                    await task
                except BaseException:
                    pass

        burst, burst_batches, lone, lone_s, staggered, stag_batches = run_async(scenario, timeout=30)
        assert list(burst) == [i * 10 for i in range(10)], f"results routed to the wrong callers: {burst}"
        sizes = [len(c) for c in burst_batches]
        assert sum(sizes) == 10, f"10 requests should be processed exactly once, batches were {sizes}"
        assert max(sizes) <= 4, f"a batch exceeded max_batch_size=4: sizes {sizes}"
        assert len(sizes) <= 4, f"10 concurrent requests should form ~3 batches of <= 4, got {len(sizes)}: {sizes}"
        assert lone == 990, f"a single request got {lone}, expected 990"
        assert lone_s < 0.06 + 0.4, f"a lone request waited {lone_s * 1e3:.0f} ms; max_wait_ms is 60"
        assert list(staggered) == [1000, 1010, 1020], f"staggered results wrong: {staggered}"
        assert [sorted(c) for c in stag_batches] == [[100, 101, 102]], (
            f"3 requests arriving within 20 ms should share one batch (max_wait_ms=60); batches: {stag_batches}. "
            "Start the wait when the first item of a batch arrives, then keep collecting until the deadline")

    return run_check("Exercise 1 · DynamicBatcher", test,
                     hint="submit: put (item, future) on an asyncio.Queue and await the future. run: await the first "
                          "item, then asyncio.wait_for(queue.get(), remaining) until full or the deadline; call "
                          "process_batch once; future.set_result for each")


def ex2(cost_per_1k):
    def test():
        cases = [(100, 3.6, 1.0, 0.01), (100, 3.6, 0.5, 0.02), (10, 0.36, 1.0, 0.01), (1000, 2.0, 0.25, 2.0 / 900)]
        for rps, price, util, want in cases:
            got = cost_per_1k(rps, price, util)
            assert close(got, want, 1e-9), (f"{rps} req/s, ${price}/h, utilisation {util}: expected "
                                            f"${want:.5f} per 1k requests, got {got}")
        a, b = cost_per_1k(50, 1.0, 0.6), cost_per_1k(100, 1.0, 0.6)
        assert close(a, 2 * b, 1e-9), "doubling throughput should halve the cost per request"

    return run_check("Exercise 2 · cost_per_1k", test,
                     hint="requests per paid hour = rps x 3600 x utilization; cost per 1k = 1000 x price / that")
