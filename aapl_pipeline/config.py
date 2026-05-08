"""Pipeline configuration.

Every parameter the pipeline may consume lives here. The defaults match
the suppor_and_resistance setup, which is the canonical backtest layout
the project uses. Both pipelines share the same backtest layout; only
the feature calculator and the parameters that drive it differ.

Most users build a config per run, either with direct kwargs or via
PipelineConfig.from_params_dict() which accepts the original notebook
style flat dict (uppercase keys included).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass
class PipelineConfig:
    """All parameters the pipeline may consume."""

    # Data source.
    data_path: str = r"D:\data\notebooks\week-10\cleaned_apple_high_low.csv"
    cutoff_date: Optional[str] = "1990-01-01"
    week_period_freq: str = "W-SUN"

    # Rolling cross validation.
    valid_weeks: int = 52
    depth_grid: Sequence[int] = field(default_factory=lambda: (2, 3, 4, 5, 6))
    leaf_grid: Sequence[int] = field(default_factory=lambda: (2, 3, 4, 5, 6))
    thresholds_tested: Sequence[float] = field(
        default_factory=lambda: tuple(np.linspace(0.01, 0.99, 99))
    )
    fixed_tree_params: Dict[str, Any] = field(default_factory=lambda: {
        "criterion": "entropy",
        "min_samples_split": 6,
        "class_weight": "balanced",
        "random_state": 42,
    })

    # Custom validation score: heavier reward for precision, light pressure
    # on chattiness. Same shape used in both original notebooks.
    alpha_p: float = 1.0
    alpha_c: float = 0.01
    p_min: float = 0.55
    c_min: float = 0.10

    # Monte Carlo.
    n_trajectories: int = 100_000
    n_weeks: int = 100
    initial_bank: float = 100.0
    upper_thresh: float = 200.0
    lower_thresh: float = 60.0
    rng_seed: int = 42

    # Subset construction. The original final_function dict called this
    # n_subsets and the SR dict called it num_subsets; from_params_dict
    # accepts either key.
    num_subsets: int = 5
    subset_start_date: Optional[str] = "2000-01-01"

    # Statistical tests.
    uniformity_binsize: int = 104

    # Support resistance envelope. Only used when SupportResistanceFeatures
    # is in the feature calculator list.
    envelope_a: float = 1.0
    envelope_b: float = 100.0
    sr_max_iter: int = 60
    sr_tol: float = 1e-9
    sr_smooth_vals: Sequence[int] = field(default_factory=lambda: (2, 4, 12))
    sr_window_vals: Sequence[int] = field(default_factory=lambda: (4, 26, 52))

    def make_rng(self) -> np.random.Generator:
        return np.random.default_rng(self.rng_seed)

    @classmethod
    def from_params_dict(cls, params: Mapping[str, Any]) -> "PipelineConfig":
        """Build a config from the original notebook style flat dict.

        Accepts the legacy uppercase keys (VALID_WEEKS, FIXED, ENVELOPE_A
        and so on), the lowercase field names, and treats n_subsets and
        num_subsets as aliases. Unknown keys like "df" are ignored so
        old param dicts work without trimming.
        """
        legacy_to_field = {
            "VALID_WEEKS": "valid_weeks",
            "depth_grid": "depth_grid",
            "leaf_grid": "leaf_grid",
            "thresholds_tested": "thresholds_tested",
            "FIXED": "fixed_tree_params",
            "alpha_p": "alpha_p",
            "alpha_c": "alpha_c",
            "p_min": "p_min",
            "c_min": "c_min",
            "n_trajectories": "n_trajectories",
            "n_weeks": "n_weeks",
            "initial_bank": "initial_bank",
            "upper_thresh": "upper_thresh",
            "lower_thresh": "lower_thresh",
            "rng_seed": "rng_seed",
            "uniformity_binsize": "uniformity_binsize",
            # n_subsets and num_subsets are the same field.
            "n_subsets": "num_subsets",
            "num_subsets": "num_subsets",
            "cutoff_date": "cutoff_date",
            "subset_start_date": "subset_start_date",
            "ENVELOPE_A": "envelope_a",
            "ENVELOPE_B": "envelope_b",
            "SR_MAX_ITER": "sr_max_iter",
            "SR_TOL": "sr_tol",
            "SR_SMOOTH_VALS": "sr_smooth_vals",
            "SR_WINDOW_VALS": "sr_window_vals",
            "data_path": "data_path",
            "week_period_freq": "week_period_freq",
        }

        valid_field_names = {f.name for f in fields(cls)}
        kwargs: Dict[str, Any] = {}
        for k, v in params.items():
            if k in legacy_to_field:
                kwargs[legacy_to_field[k]] = v
            elif k in valid_field_names:
                kwargs[k] = v
            # Anything else (notably "df") is ignored on purpose.
        return cls(**kwargs)
