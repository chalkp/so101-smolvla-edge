"""Per-chunk latency bookkeeping shared by the engines.

No torch import, so the OpenVINO runtime stays light.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Timing:
    last_ms: float = 0.0  # latest chunk, ms
    history: list[float] = field(default_factory=list)  # recent chunks, ms, oldest first

    def add(self, ms: float) -> None:
        """Record one chunk's latency; past 1000 entries the oldest 500 are dropped."""
        self.last_ms = ms
        self.history.append(ms)
        if len(self.history) > 1000:
            del self.history[:500]

    def summary(self) -> dict:
        """Count, p50/p90/p99 and max of the recorded latencies in ms, or {} if there are none."""
        h = sorted(self.history)
        if not h:
            return {}

        def quantile(p: float) -> float:
            return h[min(len(h) - 1, int(p * (len(h) - 1)))]

        return {
            "n": len(h),
            "p50_ms": quantile(0.5),
            "p90_ms": quantile(0.9),
            "p99_ms": quantile(0.99),
            "max_ms": h[-1],
        }
