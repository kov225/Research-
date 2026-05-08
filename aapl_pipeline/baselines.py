"""Baseline traders that the model is compared against.

Each subclass replays a fixed strategy over a slice of df_final and
returns the final bank. The pipeline takes the ratio of the model bank
to the baseline bank, averaged across subsets, which is what shows up
in the report under baseline_comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Baseline(ABC):
    """Parent class for naive trader baselines."""

    def __init__(self, initial_bank: float, upper_thresh: float, lower_thresh: float) -> None:
        self.initial_bank = initial_bank
        self.upper_thresh = upper_thresh
        self.lower_thresh = lower_thresh

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, sub: pd.DataFrame, rng: np.random.Generator) -> float:
        ...

    def _stop_hit(self, bank: float) -> bool:
        return bank >= self.upper_thresh or bank <= self.lower_thresh


class AlwaysTradeBaseline(Baseline):
    """Buy Tuesday, sell Thursday, every single week, no exceptions."""

    @property
    def name(self) -> str:
        return "vs_always_trade"

    def run(self, sub: pd.DataFrame, rng: np.random.Generator) -> float:
        bank = self.initial_bank
        for r in sub["thu_tue"]:
            bank *= r
            if self._stop_hit(bank):
                break
        return bank


class RandomTraderBaseline(Baseline):
    """Trades with a coin whose probability matches the model chattiness."""

    @property
    def name(self) -> str:
        return "vs_random_trader"

    def run(self, sub: pd.DataFrame, rng: np.random.Generator) -> float:
        if len(sub) == 0:
            return self.initial_bank
        trade_prob = len(sub.loc[sub["Outcome"].isin(["TP", "FP"])]) / len(sub)
        bank = self.initial_bank
        for r in sub["thu_tue"]:
            if rng.random() < trade_prob:
                bank *= r
            if self._stop_hit(bank):
                break
        return bank


class AlternateWeekBaseline(Baseline):
    """Trades every other week, starting from the first."""

    @property
    def name(self) -> str:
        return "vs_alternate_trader"

    def run(self, sub: pd.DataFrame, rng: np.random.Generator) -> float:
        bank = self.initial_bank
        for i, r in enumerate(sub["thu_tue"]):
            if i % 2 == 0:
                bank *= r
            if self._stop_hit(bank):
                break
        return bank


class WeightedCoinBaseline(Baseline):
    """Coin probability equals the historical good week rate of the slice."""

    @property
    def name(self) -> str:
        return "vs_weighted_coin"

    def run(self, sub: pd.DataFrame, rng: np.random.Generator) -> float:
        if len(sub) == 0:
            return self.initial_bank
        good_rate = float((sub["thu_tue"] > 1.0).mean())
        bank = self.initial_bank
        for r in sub["thu_tue"]:
            if rng.random() < good_rate:
                bank *= r
            if self._stop_hit(bank):
                break
        return bank
