"""Builds the weekly Tue/Thu trading table that every feature relies on.

Both the normalized feature pipeline and the support resistance pipeline
need exactly the same weekly skeleton: tue_open, thu_open, the thu/tue
multiplier and the binary week_type label. Putting it here once stops the
two feature classes from drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class WeeklyDataset:
    """Builds a per week dataframe of buy and sell open prices.

    Attributes after build():
        weekly: dataframe indexed by the W-SUN period
        df: the daily dataframe with weekday and week columns added
    """

    def __init__(self, week_period_freq: str = "W-SUN") -> None:
        self.week_period_freq = week_period_freq
        self.weekly: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None

    def build(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Annotate the daily frame and aggregate to a weekly table.

        Returns the weekly dataframe and stores both frames on the instance
        so that downstream feature calculators can reuse them.
        """
        df = daily_df.copy()
        df["DATE"] = pd.to_datetime(df["DATE"])
        df = df.sort_values("DATE").reset_index(drop=True)
        df["weekday"] = df["DATE"].dt.weekday
        df["week"] = df["DATE"].dt.to_period(self.week_period_freq)

        tue_open = (
            df.loc[df["weekday"] == 1]
              .groupby("week")["OPEN"]
              .first()
              .rename("tue_open")
        )
        thu_open = (
            df.loc[df["weekday"] == 3]
              .groupby("week")["OPEN"]
              .first()
              .rename("thu_open")
        )

        weekly = pd.concat([tue_open, thu_open], axis=1)
        weekly["thu/tue"] = weekly["thu_open"] / weekly["tue_open"]
        weekly["net%"] = (weekly["thu/tue"] - 1.0) * 100.0
        weekly["week_type"] = (weekly["thu/tue"] > 1.0).astype(int)

        self.df = df
        self.weekly = weekly
        return weekly
