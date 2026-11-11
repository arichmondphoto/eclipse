"""Running independent work on several processor cores at once.

NumPy and SciPy release Python's lock while they compute, so plain threads use several cores
(on 2 cores SciPy's filters ran 1.7-2x faster). Nothing here changes a result: work is only done
side by side, and anything that must happen in order (adding frames into a stack) still does.

``TOTALITY_THREADS`` in the environment sets the number of threads (1: one at a time, as before
10 Nov 2026); by default every core. Memory decides too: parallel work is held back so it stays
within half the computer's memory (:func:`memory_budget`).
"""

from __future__ import annotations

import os
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, Iterator, Optional, Sequence


def threads() -> int:
    """How many threads to use: ``TOTALITY_THREADS`` if set, otherwise every core."""
    env = os.environ.get("TOTALITY_THREADS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def memory_bytes() -> int:
    """The computer's physical memory (8 GB if it cannot be read)."""
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 8 * 1024**3


def memory_budget() -> int:
    """Memory that work running side by side may take: half the computer's memory."""
    return memory_bytes() // 2


def describe() -> str:
    """For the log: what this computer gives the work."""
    return f"{threads()} thread(s), up to {memory_budget() / 1024**3:.0f} GB for work done side by side"


class _Budget:
    """Lets a task start only while the tasks running together stay within ``total`` bytes; a task
    larger than the whole budget still runs, on its own."""

    def __init__(self, total: int) -> None:
        self.total, self.used = int(total), 0
        self.cond = threading.Condition()

    def acquire(self, n: int) -> int:
        n = int(n)
        with self.cond:
            while self.used > 0 and self.used + n > self.total:
                self.cond.wait()
            self.used += n
        return n

    def release(self, n: int) -> None:
        with self.cond:
            self.used -= n
            self.cond.notify_all()


def run_all(tasks: Sequence[tuple], budget: Optional[int] = None, workers: Optional[int] = None) -> list:
    """Run ``tasks``, each ``(fn, bytes_needed)``, side by side within the memory ``budget``
    (default :func:`memory_budget`) on up to ``workers`` threads; returns their results in order.
    An error in any task is raised here once the others have finished."""
    n = min(len(tasks), workers or threads())
    if n <= 1:
        return [fn() for fn, _need in tasks]
    gate = _Budget(memory_budget() if budget is None else budget)

    def one(task):
        fn, need = task
        got = gate.acquire(need)
        try:
            return fn()
        finally:
            gate.release(got)

    with ThreadPoolExecutor(n) as ex:
        futures = [ex.submit(one, t) for t in tasks]
        return [f.result() for f in futures]


def ahead(items: Iterable, fn: Callable, bytes_each: int = 0, workers: Optional[int] = None) -> Iterator:
    """``fn(item)`` for each item, in order, with the next items already being worked on in the
    background: as many as there are threads, and as their memory (``bytes_each``) fits in the
    budget. For frames that are prepared independently but added up in order."""
    items = list(items)
    n = workers or threads()
    if bytes_each > 0:
        n = min(n, max(1, memory_budget() // int(bytes_each)))
    if n <= 1 or len(items) <= 1:
        for it in items:
            yield fn(it)
        return
    with ThreadPoolExecutor(n) as ex:
        queue: deque = deque()
        it = iter(items)
        for item in it:
            queue.append(ex.submit(fn, item))
            if len(queue) >= n:
                break
        end = object()
        while queue:
            result = queue.popleft().result()
            nxt = next(it, end)
            if nxt is not end:
                queue.append(ex.submit(fn, nxt))
            yield result
