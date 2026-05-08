"""Loads the cleaned daily AAPL CSV.

The CSV layout assumed here is the one produced by the duplicate cleaning
notebook: columns DATE, weekday, OPEN, CLOSE, VOL, HIGH, LOW with weekend
rows already dropped and US holidays forward filled.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd


class DataLoader:
    """Reads a cleaned daily price file into a tidy dataframe.

    The default path matches the file produced by the cleaning notebook.
    Pass a different path to load a separate ticker or a fresh extract.
    """

    def __init__(
        self,
        path: str = r"D:\data\notebooks\week-10\cleaned_apple_high_low.csv",
        date_column: str = "DATE",
    ) -> None:
        self.path = path
        self.date_column = date_column

    def load(self, cutoff_date: Optional[str] = None) -> pd.DataFrame:
        """Read the CSV, parse dates, sort, and optionally filter early years.

        Sorting by date matters because the rolling backtest relies on the
        natural order of rows.
        """
        df = pd.read_csv(self.path)
        df[self.date_column] = pd.to_datetime(df[self.date_column], errors="coerce")
        df = df.sort_values(self.date_column).reset_index(drop=True)

        if cutoff_date is not None:
            cutoff_dt = pd.to_datetime(cutoff_date)
            df = df[df[self.date_column] >= cutoff_dt].reset_index(drop=True)

        return df
