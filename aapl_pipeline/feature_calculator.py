"""Feature calculators.

The contract is intentionally narrow. Each subclass receives the daily and
weekly dataframes and returns its own dataframe indexed by the same week
period. The pipeline concatenates the outputs of every registered
calculator and that single concatenated frame becomes the model input.

Adding a new family of features means writing one new subclass and
appending an instance to the pipeline. Nothing else needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence

import numpy as np
import pandas as pd


class FeatureCalculator(ABC):
    """Parent class for every feature family used by the strategy.

    Subclasses must override compute() and feature_names. The pipeline only
    talks to this interface, which keeps the rest of the code stable when a
    new family of features is added.
    """

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        """Names of the columns this calculator produces."""

    @abstractmethod
    def compute(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
        """Return a dataframe indexed by the weekly period with feature columns."""


class TueThuNormalizedFeatures(FeatureCalculator):
    """Causal z scores of three opens, the way final_function_tue_thu does it.

    The expanding mean and std are shifted by one so the score for any row
    only uses information that was available before that row. That keeps the
    rolling backtest honest.
    """

    def __init__(self) -> None:
        self._features = ["Norm_PrevThu_Open", "Norm_PrevFri_Open", "Norm_Tue_Open"]

    @property
    def feature_names(self) -> List[str]:
        return list(self._features)

    def compute(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()

        df["normalized_open"] = (
            (df["OPEN"] - df["OPEN"].expanding().mean().shift(1))
            / df["OPEN"].expanding().std(ddof=0).shift(1)
        )

        norm_tue_open = (
            df.loc[df["weekday"] == 1]
              .set_index("week")["normalized_open"]
              .rename("Norm_Tue_Open")
        )
        norm_prev_thu_open = (
            df.loc[df["weekday"] == 3]
              .set_index("week")["normalized_open"]
              .rename("Norm_PrevThu_Open")
              .shift(1)
        )
        norm_prev_fri_open = (
            df.loc[df["weekday"] == 4]
              .set_index("week")["normalized_open"]
              .rename("Norm_PrevFri_Open")
              .shift(1)
        )

        out = pd.concat([norm_tue_open, norm_prev_thu_open, norm_prev_fri_open], axis=1)
        return out.reindex(weekly_df.index)


class SupportResistanceFeatures(FeatureCalculator):
    """Envelope based support and resistance features.

    For every (smoothing window, lookback weeks) pair we fit two weighted
    lines through the smoothed Monday midprices: one biased toward the
    bottom of the cloud (support) and one biased toward the top
    (resistance). The feature is the position of Tuesday open inside that
    envelope. Values near 0 mean Tuesday is sitting at support, values near
    1 mean it is sitting at resistance.

    The penalty weights envelope_a and envelope_b control how aggressively
    the line clings to the lower or upper edge.
    """

    def __init__(
        self,
        smooth_vals: Sequence[int] = (2, 4, 12),
        window_vals: Sequence[int] = (4, 26, 52),
        envelope_a: float = 1.0,
        envelope_b: float = 100.0,
        max_iter: int = 60,
        tol: float = 1e-9,
    ) -> None:
        self.smooth_vals = tuple(smooth_vals)
        self.window_vals = tuple(window_vals)
        self.envelope_a = envelope_a
        self.envelope_b = envelope_b
        self.max_iter = max_iter
        self.tol = tol

    @property
    def feature_names(self) -> List[str]:
        return [f"SR_{sw}_{ww}" for sw in self.smooth_vals for ww in self.window_vals]

    def _fit_line_weighted(
        self,
        t: np.ndarray,
        y: np.ndarray,
        g: np.ndarray,
        a: float,
        b: float,
    ) -> tuple[float, float]:
        """Iteratively reweighted line fit.

        We start with an ordinary least squares slope and intercept, then at
        each iteration we down weight points on the wrong side of the line.
        Setting a small and b large pulls the line toward the lower edge of
        the cloud. Swapping them pulls it toward the upper edge.
        """
        X = np.column_stack([t, np.ones_like(t)])
        m, c = np.linalg.lstsq(X, y, rcond=None)[0]

        for _ in range(self.max_iter):
            r = y - (m * t + c)
            k = np.where(r > 0, a, np.where(r < 0, b, 0.0))
            w = g * k

            S_tt = np.sum(w * t * t)
            S_t = np.sum(w * t)
            S_1 = np.sum(w)
            R_t = np.sum(w * t * y)
            R_1 = np.sum(w * y)

            A = np.array([[S_tt, S_t], [S_t, S_1]])
            B = np.array([R_t, R_1])

            if abs(np.linalg.det(A)) < 1e-12:
                break

            m_new, c_new = np.linalg.solve(A, B)
            if abs(m_new - m) + abs(c_new - c) < self.tol:
                m, c = m_new, c_new
                break
            m, c = m_new, c_new

        return float(m), float(c)

    def _build_envelope(
        self,
        mon_df: pd.DataFrame,
        smooth_window: int,
        window_weeks: int,
        a: float,
        b: float,
    ) -> pd.Series:
        """Build the envelope value at each Monday by fitting on a trailing window."""
        m = mon_df.copy()
        m["mid_smooth"] = m["mid"].rolling(smooth_window, min_periods=smooth_window).mean()
        m = m.dropna(subset=["mid_smooth"]).reset_index(drop=True)

        m["t"] = np.arange(len(m), dtype=float)
        t_all = m["t"].to_numpy()
        y_all = m["mid_smooth"].to_numpy()
        env = np.full(len(m), np.nan)

        for i in range(len(m)):
            T = t_all[i]
            start = max(0, i - window_weeks)
            idx = slice(start, i + 1)

            t_win = t_all[idx]
            y_win = y_all[idx]
            g = (t_win - (T - window_weeks)) / window_weeks

            m_fit, c_fit = self._fit_line_weighted(t_win, y_win, g, a, b)
            env[i] = m_fit * T + c_fit

        m["envelope"] = env
        m["week"] = m["DATE"].dt.to_period("W-SUN")
        return m.set_index("week")["envelope"]

    def compute(self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
        df = daily_df.copy()
        df["mid"] = (df["HIGH"] + df["LOW"]) / 2.0
        mon_raw = df[df["weekday"] == 0].copy().reset_index(drop=True)

        out = pd.DataFrame(index=weekly_df.index)

        for sw in self.smooth_vals:
            for ww in self.window_vals:
                col = f"SR_{sw}_{ww}"

                support = self._build_envelope(mon_raw, sw, ww, self.envelope_a, self.envelope_b)
                resistance = self._build_envelope(mon_raw, sw, ww, self.envelope_b, self.envelope_a)

                support_aligned = support.reindex(weekly_df.index)
                resistance_aligned = resistance.reindex(weekly_df.index)

                denom = resistance_aligned - support_aligned
                pos = (weekly_df["tue_open"] - support_aligned) / denom

                # A non positive width means the resistance fell below
                # support, which is geometrically meaningless.
                bad = denom <= 0
                pos = pos.where(~bad, np.nan)

                out[col] = pos

        return out
