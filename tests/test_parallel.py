"""Work side by side on several cores (list item 9, Nov 2026): the same results as one at a time."""

import threading
import time

import numpy as np
import pytest

from tse_core import enhance as E
from tse_core import parallel


def test_ahead_keeps_order_and_reads_ahead(monkeypatch):
    monkeypatch.setenv("TOTALITY_THREADS", "3")
    started = []

    def work(i):
        started.append(i)
        time.sleep(0.01 * (5 - i % 5))  # later items finish first
        return i * i

    got = []
    for y in parallel.ahead(range(10), work):
        got.append(y)
    assert got == [i * i for i in range(10)]
    assert sorted(started) == list(range(10))


def test_run_all_keeps_within_the_memory_budget(monkeypatch):
    monkeypatch.setenv("TOTALITY_THREADS", "8")
    lock, now, most = threading.Lock(), [0], [0]

    def task():
        with lock:
            now[0] += 1
            most[0] = max(most[0], now[0])
        time.sleep(0.02)
        with lock:
            now[0] -= 1
        return 1

    # 8 threads, but only two tasks of 40 fit in a budget of 100 at once
    assert parallel.run_all([(task, 40)] * 6, budget=100) == [1] * 6
    assert most[0] == 2
    # a task larger than the whole budget still runs, alone
    most[0] = 0
    assert parallel.run_all([(task, 500)] * 3, budget=100) == [1] * 3 and most[0] == 1


def test_errors_come_through():
    def bad():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        parallel.run_all([(lambda: 1, 1), (bad, 1)], budget=10, workers=2)


@pytest.mark.parametrize("n", ["1", "4"])
def test_threads_setting(monkeypatch, n):
    monkeypatch.setenv("TOTALITY_THREADS", n)
    assert parallel.threads() == int(n)


def test_stack_is_the_same_whatever_the_threads(monkeypatch):
    from test_stack import _synthetic_stack

    out = {}
    for n in ("1", "4"):
        monkeypatch.setenv("TOTALITY_THREADS", n)
        _res, hdr, weight, *_ = _synthetic_stack(noise=True, errors={1: 0.05})
        out[n] = (hdr, weight)
    assert np.array_equal(out["1"][0], out["4"][0]) and np.array_equal(out["1"][1], out["4"][1])


def test_enhance_layers_are_the_same_whatever_the_threads(monkeypatch):
    from test_achf import _real_frame

    p = E.EnhanceParams(layers=("natural", "nrgf", "fnrgf_n", "adaptive", "wow", "achf"),
                        blend={"natural": 1.0, "fnrgf_n": 0.5, "adaptive": 0.4, "wow": 0.3},
                        blend_modes={"adaptive": "average", "wow": "linear"})
    out = {}
    for n in ("1", "4"):
        monkeypatch.setenv("TOTALITY_THREADS", n)
        out[n] = E.render(_real_frame(p), p)
    assert set(out["1"]) == set(out["4"])
    for k in out["1"]:
        assert np.array_equal(out["1"][k], out["4"][k]), k
