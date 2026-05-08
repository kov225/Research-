"""Report comparison hierarchy.

Two report dicts are produced by two pipeline runs (different features,
different model, different anything). Comparator subclasses turn a pair
of those dicts into a verdict on which model is better.

The current implementation, WeightedVoteComparator, casts one weighted
vote per scalar metric in the registry. Future variants like a relative
magnitude scorer would subclass ReportComparator and override
compare(...). The orchestrator does not change.
"""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from aapl_pipeline.performance_metric import MetricRegistry, PerformanceMetric


def _displayable(value: Any) -> Any:
    """Compact representation of an array or list so display tables stay readable."""
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, list) and len(value) > 6:
        return f"list(len={len(value)})"
    return value


@dataclass
class ComparisonResult:
    """What every ReportComparator returns."""
    name_a: str
    name_b: str
    rows: List[Dict[str, Any]] = field(default_factory=list)
    raw_count: Dict[str, int] = field(default_factory=dict)
    weighted_total: Dict[str, float] = field(default_factory=dict)
    verdict: str = "tie"

    def to_dataframe(self) -> pd.DataFrame:
        """Per metric breakdown as a dataframe.

        Array values are condensed to shape descriptors so the display
        does not blow up on Monte Carlo rows. Original values are still
        available on `self.rows`.
        """
        cleaned = []
        for row in self.rows:
            cleaned.append({k: _displayable(v) for k, v in row.items()})
        return pd.DataFrame(cleaned)

    def voting_rows(self) -> pd.DataFrame:
        """Only the rows that actually contributed to the verdict."""
        df = self.to_dataframe()
        if df.empty:
            return df
        return df[df["weight"] > 0].reset_index(drop=True)

    def __repr__(self) -> str:
        a, b = self.name_a, self.name_b
        lines = [
            f"ComparisonResult(verdict={self.verdict})",
            f"  raw_count:      {a}={self.raw_count.get(a, 0)}, "
            f"{b}={self.raw_count.get(b, 0)}, "
            f"ties={self.raw_count.get('ties', 0)}",
            f"  weighted_total: {a}={self.weighted_total.get(a, 0.0):.2f}, "
            f"{b}={self.weighted_total.get(b, 0.0):.2f}",
        ]
        return "\n".join(lines)


class ReportComparator(ABC):
    """Parent class for any rule that compares two report dicts.

    Subclasses receive a registry so they can iterate metrics in the
    same order they were computed and use each metric's `direction` and
    `(group, name)` location.
    """

    def __init__(self, registry: MetricRegistry) -> None:
        self.registry = registry

    @abstractmethod
    def compare(
        self,
        report_a: Dict[str, Any],
        report_b: Dict[str, Any],
        name_a: str = "A",
        name_b: str = "B",
    ) -> ComparisonResult:
        ...

    def print_summary(self, result: ComparisonResult) -> None:
        """Default human readable rendering. Subclasses can override."""
        df = result.to_dataframe()
        if not df.empty:
            with pd.option_context("display.max_rows", None, "display.width", 200):
                print(df.to_string(index=False))
            print()
        print(repr(result))


class WeightedVoteComparator(ReportComparator):
    """Each scalar metric casts `weight` votes for whichever side wins.

    Default weight is 1.0 for any metric whose direction is "maximize"
    or "minimize", and 0.0 for anything "neutral" (raw arrays,
    descriptive stats, statistical test outputs). Pass a `weights` dict
    keyed by "group.name" to override.
    """

    def __init__(
        self,
        registry: MetricRegistry,
        weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__(registry)
        self.weights = dict(weights or {})

    def _key(self, metric: PerformanceMetric) -> str:
        return f"{metric.group}.{metric.name}"

    def _weight_for(self, metric: PerformanceMetric) -> float:
        key = self._key(metric)
        if key in self.weights:
            return float(self.weights[key])
        return 0.0 if metric.direction == "neutral" else 1.0

    def _decide(self, val_a: Any, val_b: Any, direction: str, name_a: str, name_b: str) -> str:
        # Only compare comparable real numbers. Booleans are excluded so
        # accidental True/False values do not silently rank.
        is_scalar = (
            isinstance(val_a, numbers.Real) and not isinstance(val_a, bool)
            and isinstance(val_b, numbers.Real) and not isinstance(val_b, bool)
        )
        if not is_scalar:
            return "n/a"
        if np.isnan(val_a) or np.isnan(val_b):
            return "n/a"
        if val_a == val_b:
            return "tie"
        if direction == "maximize":
            return name_a if val_a > val_b else name_b
        if direction == "minimize":
            return name_a if val_a < val_b else name_b
        return "n/a"

    def compare(
        self,
        report_a: Dict[str, Any],
        report_b: Dict[str, Any],
        name_a: str = "A",
        name_b: str = "B",
    ) -> ComparisonResult:
        rows: List[Dict[str, Any]] = []
        raw_a = raw_b = ties = 0
        wt_a = wt_b = 0.0

        for metric in self.registry:
            val_a = report_a.get(metric.group, {}).get(metric.name)
            val_b = report_b.get(metric.group, {}).get(metric.name)
            weight = self._weight_for(metric)

            winner = self._decide(val_a, val_b, metric.direction, name_a, name_b)

            # Track raw counts only when there is a real winner.
            if winner == name_a:
                raw_a += 1
                wt_a += weight
            elif winner == name_b:
                raw_b += 1
                wt_b += weight
            elif winner == "tie":
                ties += 1

            rows.append({
                "metric": self._key(metric),
                "direction": metric.direction,
                name_a: val_a,
                name_b: val_b,
                "weight": weight,
                "winner": winner,
            })

        if wt_a > wt_b:
            verdict = name_a
        elif wt_b > wt_a:
            verdict = name_b
        else:
            verdict = "tie"

        return ComparisonResult(
            name_a=name_a,
            name_b=name_b,
            rows=rows,
            raw_count={name_a: raw_a, name_b: raw_b, "ties": ties},
            weighted_total={name_a: wt_a, name_b: wt_b},
            verdict=verdict,
        )
