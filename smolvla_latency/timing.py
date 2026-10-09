"""Region timing for one policy call, synchronised so GPU work is charged to the right region."""

import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import torch


class RegionTimer:
    """Accumulates wall time per named region within a call, then stores one row per call.

    Every region boundary calls ``torch.cuda.synchronize()``. That makes per-region numbers
    honest at batch size 1 but adds a little overhead, so callers also measure an
    uninstrumented end-to-end time and report the difference.
    """

    def __init__(self, device: torch.device):
        self.cuda = device.type == "cuda"
        self.rows: list[dict[str, float]] = []
        self._cur: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self.enabled = True

    def sync(self) -> None:
        if self.cuda:
            torch.cuda.synchronize()

    @contextmanager
    def region(self, name: str):
        if not self.enabled:
            yield
            return
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self._cur[name] += (time.perf_counter() - t0) * 1e3
            self._counts[name] += 1

    def end_call(self, keep: bool) -> dict[str, float]:
        row = dict(self._cur)
        row.update({f"n_{k}": float(v) for k, v in self._counts.items()})
        if keep:
            self.rows.append(row)
        self._cur = defaultdict(float)
        self._counts = defaultdict(int)
        return row


def summarize(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {}
    return {
        "mean": float(a.mean()),
        "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }
