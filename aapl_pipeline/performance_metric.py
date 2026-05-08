"""Performance metric hierarchy and registry.

Every number that ends up in the report dict is one PerformanceMetric
subclass. Each subclass returns one value from compute(), declares the
group and name it lives under in the report, and declares a direction
("maximize", "minimize", or "neutral") that downstream tooling like the
ReportComparator uses to decide which side wins.

To add a new metric: subclass, set group, name, direction, implement
compute(), and call registry.register on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp
from statsmodels.sandbox.stats.runs import runstest_1samp


@dataclass
class MetricContext:
    """All the data a metric might need to compute itself.

    The pipeline fills this in once and hands the same object to every
    metric. Metrics pick out only the fields they care about.
    """
    df_final: pd.DataFrame
    confusion: Dict[str, int]
    initial_bank: float
    sim_all: np.ndarray
    actual_balances: np.ndarray
    null_percentiles: List[float]
    future_sims: np.ndarray
    upper_thresh: float
    lower_thresh: float
    uniformity_binsize: int
    baseline_ratios: Dict[str, float] = field(default_factory=dict)


class PerformanceMetric(ABC):
    """Parent class for every value in the report card.

    Subclasses must set group, name, direction, and implement compute().
    direction is one of:
      - "maximize"  bigger is better
      - "minimize"  smaller is better
      - "neutral"   no opinion (raw arrays, dof, descriptive stats)
    """

    direction: str = "neutral"

    @property
    @abstractmethod
    def group(self) -> str:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def compute(self, ctx: MetricContext) -> Any:
        ...


# Internal metrics: confusion based diagnostics on the rolling backtest.

class PrecisionOverall(PerformanceMetric):
    group = "internal_metrics"
    name = "precision_overall"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        tp, fp = ctx.confusion["TP"], ctx.confusion["FP"]
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0


class ChattinessOverall(PerformanceMetric):
    group = "internal_metrics"
    name = "chattiness_overall"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        tp, fp, fn = ctx.confusion["TP"], ctx.confusion["FP"], ctx.confusion["FN"]
        return (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0.0


class CorrectnessRate(PerformanceMetric):
    group = "internal_metrics"
    name = "correctness_rate"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        c = ctx.confusion
        total = c["TP"] + c["TN"] + c["FP"] + c["FN"]
        return (c["TP"] + c["TN"]) / total if total > 0 else 0.0


class TradeFrequency(PerformanceMetric):
    group = "internal_metrics"
    name = "trade_frequency"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        df = ctx.df_final
        if len(df) == 0:
            return 0.0
        traded = (df["Outcome"].isin(["TP", "FP"])).sum()
        return float(traded) / len(df)


class MistakeAsymmetry(PerformanceMetric):
    """Average winning move plus average losing move, in percent.

    Reads as: when the model trades, do its hits move the bank by more
    than its misses move it the other way.
    """
    group = "internal_metrics"
    name = "mistake_asymmetry_%"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        df = ctx.df_final
        tp_vals = df.loc[df["Outcome"] == "TP", "thu_tue"].values
        fp_vals = df.loc[df["Outcome"] == "FP", "thu_tue"].values
        if len(tp_vals) == 0 or len(fp_vals) == 0:
            return float("nan")
        tp_pct = (tp_vals - 1) * 100
        fp_pct = (fp_vals - 1) * 100
        return float(tp_pct.mean() + fp_pct.mean())


class LongestStreak(PerformanceMetric):
    """Longest run of a single outcome label in chronological order."""
    group = "internal_metrics"

    def __init__(self, label: str) -> None:
        self.label = label
        # Long FP streaks are bad. Long TP streaks are not directly a
        # quality signal, so leave them neutral.
        self.direction = "minimize" if label == "FP" else "neutral"

    @property
    def name(self) -> str:
        return f"longest_{self.label}_streak"

    def compute(self, ctx: MetricContext) -> int:
        best = cur = 0
        for x in ctx.df_final["Outcome"]:
            if x == self.label:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)


class FPRateWhenPredictedPositive(PerformanceMetric):
    group = "internal_metrics"
    name = "%FP_when_predicted_positive"
    direction = "minimize"

    def compute(self, ctx: MetricContext) -> float:
        tp, fp = ctx.confusion["TP"], ctx.confusion["FP"]
        return fp / (tp + fp) if (tp + fp) > 0 else 0.0


# Historical Monte Carlo metrics.

class SimulatedMean(PerformanceMetric):
    group = "historical_mc"
    name = "simulated_mean"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        if len(ctx.sim_all) == 0:
            return float(ctx.initial_bank)
        return float(np.mean(ctx.sim_all))


class SimulatedMedian(PerformanceMetric):
    group = "historical_mc"
    name = "simulated_median"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        if len(ctx.sim_all) == 0:
            return float(ctx.initial_bank)
        return float(np.median(ctx.sim_all))


class KSDistance(PerformanceMetric):
    group = "historical_mc"
    name = "ks_distance"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        if len(ctx.actual_balances) < 1 or len(ctx.sim_all) < 1:
            return float("nan")
        d, _ = ks_2samp(ctx.actual_balances, ctx.sim_all)
        return float(d)


class KSPValue(PerformanceMetric):
    group = "historical_mc"
    name = "ks_p_value"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        if len(ctx.actual_balances) < 1 or len(ctx.sim_all) < 1:
            return float("nan")
        _, p = ks_2samp(ctx.actual_balances, ctx.sim_all)
        return float(p)


class AverageNullPercentile(PerformanceMetric):
    """Mean rank of actual balance inside its simulated null distribution."""
    group = "historical_mc"
    name = "average_null_percentile"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        if len(ctx.null_percentiles) == 0:
            return 0.5
        return float(np.mean(ctx.null_percentiles))


class HistoricalSimulations(PerformanceMetric):
    """Concatenated bank trajectories from every historical MC subset."""
    group = "historical_mc"
    name = "simulations"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> np.ndarray:
        return ctx.sim_all


class HistoricalActuals(PerformanceMetric):
    """Actual model bank balance at the end of each historical subset."""
    group = "historical_mc"
    name = "actual_balances"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> np.ndarray:
        return ctx.actual_balances


class HistoricalNullPercentiles(PerformanceMetric):
    """Per subset rank of the actual balance inside the simulated null."""
    group = "historical_mc"
    name = "null_percentiles"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> List[float]:
        return list(ctx.null_percentiles)


# Future Monte Carlo metrics.

class FutureMean(PerformanceMetric):
    group = "future_mc"
    name = "future_mean"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        return float(np.mean(ctx.future_sims))


class FutureMedian(PerformanceMetric):
    group = "future_mc"
    name = "future_median"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        return float(np.median(ctx.future_sims))


class ProbAboveInitial(PerformanceMetric):
    group = "future_mc"
    name = "prob_above_initial"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        return float(np.mean(ctx.future_sims > ctx.initial_bank))


class ProbSuccess(PerformanceMetric):
    group = "future_mc"
    name = "prob_success"
    direction = "maximize"

    def compute(self, ctx: MetricContext) -> float:
        return float(np.mean(ctx.future_sims >= ctx.upper_thresh))


class ProbFailure(PerformanceMetric):
    group = "future_mc"
    name = "prob_failure"
    direction = "minimize"

    def compute(self, ctx: MetricContext) -> float:
        return float(np.mean(ctx.future_sims <= ctx.lower_thresh))


class ProbUncertain(PerformanceMetric):
    """Probability the bank ends inside the do nothing band."""
    group = "future_mc"
    name = "prob_uncertain"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        success = float(np.mean(ctx.future_sims >= ctx.upper_thresh))
        failure = float(np.mean(ctx.future_sims <= ctx.lower_thresh))
        return float(1.0 - success - failure)


class FutureSimulations(PerformanceMetric):
    """Forward looking bank trajectories sampled from all of df_final."""
    group = "future_mc"
    name = "simulations"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> np.ndarray:
        return ctx.future_sims


# Statistical tests. Each one is a single scalar metric so each entry in
# the report card is one number, in line with the rule that a metric
# returns one value of the metric we chose.

class RunsTestZ(PerformanceMetric):
    """z statistic from the Wald Wolfowitz runs test on the correctness sequence."""
    group = "randomness_test"
    name = "z"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        z, _ = runstest_1samp(ctx.df_final["correct"], correction=False)
        return float(z)


class RunsTestPValue(PerformanceMetric):
    """p value from the Wald Wolfowitz runs test."""
    group = "randomness_test"
    name = "p_value"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        _, p = runstest_1samp(ctx.df_final["correct"], correction=False)
        return float(p)


class ChiSquareStatistic(PerformanceMetric):
    """Chi square statistic for uniformity of correctness across time bins."""
    group = "uniformity_test"
    name = "chi2"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        df = ctx.df_final.copy()
        df["chunk"] = df.index // ctx.uniformity_binsize
        chi2, _, _, _ = chi2_contingency(pd.crosstab(df["chunk"], df["correct"]))
        return float(chi2)


class ChiSquarePValue(PerformanceMetric):
    """p value of the uniformity chi square test."""
    group = "uniformity_test"
    name = "p_value"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> float:
        df = ctx.df_final.copy()
        df["chunk"] = df.index // ctx.uniformity_binsize
        _, p, _, _ = chi2_contingency(pd.crosstab(df["chunk"], df["correct"]))
        return float(p)


class ChiSquareDoF(PerformanceMetric):
    """Degrees of freedom of the uniformity chi square test."""
    group = "uniformity_test"
    name = "dof"
    direction = "neutral"

    def compute(self, ctx: MetricContext) -> int:
        df = ctx.df_final.copy()
        df["chunk"] = df.index // ctx.uniformity_binsize
        _, _, dof, _ = chi2_contingency(pd.crosstab(df["chunk"], df["correct"]))
        return int(dof)


# Baseline ratios are computed by Baseline classes inside the pipeline.
# This wrapper just exposes whichever ratios were precomputed onto ctx.

class BaselineRatio(PerformanceMetric):
    group = "baseline_comparison"
    direction = "maximize"

    def __init__(self, baseline_name: str) -> None:
        self.baseline_name = baseline_name

    @property
    def name(self) -> str:
        return self.baseline_name

    def compute(self, ctx: MetricContext) -> float:
        return float(ctx.baseline_ratios.get(self.baseline_name, float("nan")))


# Registry.

class MetricRegistry:
    """Holds metric instances and runs them all in one pass.

    Output is a nested dict keyed by group then name. Every metric returns
    one value, so each (group, name) cell in the report carries one
    number (or one array, for the raw MC entries).
    """

    def __init__(self) -> None:
        self._metrics: List[PerformanceMetric] = []

    def register(self, metric: PerformanceMetric) -> None:
        self._metrics.append(metric)

    def __iadd__(self, metric: PerformanceMetric) -> "MetricRegistry":
        self.register(metric)
        return self

    def __iter__(self):
        return iter(self._metrics)

    def __len__(self) -> int:
        return len(self._metrics)

    def run_all(self, ctx: MetricContext, round_to: Optional[int] = 2) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        for metric in self._metrics:
            value = metric.compute(ctx)
            report.setdefault(metric.group, {})[metric.name] = value
        if round_to is not None:
            report = _round_floats(report, round_to)
        return report


def _round_floats(obj: Any, ndigits: int) -> Any:
    # ndarrays carry their own dtype and are usually large MC outputs we
    # want to leave alone, so handle them before the generic float check.
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: _round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(x, ndigits) for x in obj]
    return obj


def default_registry() -> MetricRegistry:
    """Registry preloaded with every metric the project ships with."""
    reg = MetricRegistry()

    reg.register(SimulatedMean())
    reg.register(SimulatedMedian())
    reg.register(KSDistance())
    reg.register(KSPValue())
    reg.register(AverageNullPercentile())
    reg.register(HistoricalSimulations())
    reg.register(HistoricalActuals())
    reg.register(HistoricalNullPercentiles())

    reg.register(FutureMean())
    reg.register(FutureMedian())
    reg.register(ProbAboveInitial())
    reg.register(ProbSuccess())
    reg.register(ProbFailure())
    reg.register(ProbUncertain())
    reg.register(FutureSimulations())

    reg.register(PrecisionOverall())
    reg.register(ChattinessOverall())
    reg.register(CorrectnessRate())
    reg.register(TradeFrequency())
    reg.register(MistakeAsymmetry())
    reg.register(LongestStreak("TP"))
    reg.register(LongestStreak("FP"))
    reg.register(FPRateWhenPredictedPositive())

    for baseline_name in [
        "vs_always_trade",
        "vs_random_trader",
        "vs_alternate_trader",
        "vs_weighted_coin",
    ]:
        reg.register(BaselineRatio(baseline_name))

    reg.register(ChiSquareStatistic())
    reg.register(ChiSquarePValue())
    reg.register(ChiSquareDoF())
    reg.register(RunsTestZ())
    reg.register(RunsTestPValue())

    return reg
