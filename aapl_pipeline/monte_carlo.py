"""Monte Carlo simulation of the bank balance over time.

The block samples weekly outcomes from the historical mix of TP, FP, FN,
TN. When an outcome is TP or FP we draw a thu/tue multiplier from the
empirical distribution of the relevant pile and apply it to the bank.
FN and TN do not multiply because the strategy did not enter that week.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


class EmpiricalSampler:
    """Inverse CDF sampler over a fixed array of historical multipliers.

    Falls back to multiplier 1.0 when the history is empty, which means
    those weeks contribute nothing to the bank.
    """

    def __init__(self, values: np.ndarray, rng: np.random.Generator) -> None:
        self.rng = rng
        self.values = np.sort(np.asarray(values, dtype=float)) if len(values) else np.empty(0)
        if len(self.values) > 0:
            self.cdf = np.arange(1, len(self.values) + 1) / len(self.values)
        else:
            self.cdf = np.empty(0)

    def is_empty(self) -> bool:
        return len(self.values) == 0

    def sample_one(self) -> float:
        if self.is_empty():
            return 1.0
        u = self.rng.random()
        idx = np.searchsorted(self.cdf, u)
        idx = min(idx, len(self.values) - 1)
        return float(self.values[idx])


class MonteCarloSimulator:
    """Runs trajectories of bank balance under historical outcome mix.

    The same class drives both the historical block, where the outcome
    mix and samplers come from past windows, and the future block, where
    they come from the full backtest.
    """

    OUTCOMES = np.array(["TP", "FP", "FN", "TN"])

    def __init__(
        self,
        n_trajectories: int,
        n_weeks: int,
        initial_bank: float,
        upper_thresh: float,
        lower_thresh: float,
        rng: np.random.Generator,
    ) -> None:
        self.n_trajectories = n_trajectories
        self.n_weeks = n_weeks
        self.initial_bank = initial_bank
        self.upper_thresh = upper_thresh
        self.lower_thresh = lower_thresh
        self.rng = rng

    def run(
        self,
        outcome_probs: np.ndarray,
        tp_sampler: EmpiricalSampler,
        fp_sampler: EmpiricalSampler,
    ) -> np.ndarray:
        cdf = np.cumsum(outcome_probs)
        final = np.empty(self.n_trajectories)

        for i in range(self.n_trajectories):
            bank = self.initial_bank
            for _ in range(self.n_weeks):
                r = self.rng.random()
                idx = np.searchsorted(cdf, r)
                idx = min(idx, len(self.OUTCOMES) - 1)
                outcome = self.OUTCOMES[idx]

                if outcome == "TP":
                    bank *= tp_sampler.sample_one()
                elif outcome == "FP":
                    bank *= fp_sampler.sample_one()

                if bank >= self.upper_thresh or bank <= self.lower_thresh:
                    break
            final[i] = bank
        return final

    def run_actual(self, sub: pd.DataFrame) -> float:
        """Replay the actual model decisions on a slice of df_final."""
        bank = self.initial_bank
        for _, row in sub.iterrows():
            if row["Outcome"] in ("TP", "FP"):
                bank *= row["thu_tue"]
            if bank >= self.upper_thresh or bank <= self.lower_thresh:
                break
        return bank

    @staticmethod
    def outcome_probs_from(sub: pd.DataFrame) -> np.ndarray:
        """Outcome mix from a slice of df_final, aligned to OUTCOMES order."""
        counts = sub["Outcome"].value_counts(normalize=True)
        return np.array([counts.get(k, 0.0) for k in MonteCarloSimulator.OUTCOMES])
