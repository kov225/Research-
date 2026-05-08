"""Strategy pipeline orchestrator.

The backtest layout is the suppor_and_resistance one and it is shared by
every run: linspace subset start points after subset_start_date,
historical Monte Carlo that samples from the history strictly before
each subset (skipping the first), and a future Monte Carlo that samples
from all of df_final. The only thing that differs between runs is the
feature calculator and the inputs that drive it.

Two factory helpers reproduce the two original notebooks by changing
only the feature calculator: make_tue_thu_pipeline and
make_support_resistance_pipeline.
"""

from __future__ import annotations

from pprint import pprint
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from aapl_pipeline.baselines import (
    AlternateWeekBaseline,
    AlwaysTradeBaseline,
    Baseline,
    RandomTraderBaseline,
    WeightedCoinBaseline,
)
from aapl_pipeline.config import PipelineConfig
from aapl_pipeline.data_loader import DataLoader
from aapl_pipeline.feature_calculator import (
    FeatureCalculator,
    SupportResistanceFeatures,
    TueThuNormalizedFeatures,
)
from aapl_pipeline.modeller import DecisionTreeModeller, Modeller
from aapl_pipeline.monte_carlo import EmpiricalSampler, MonteCarloSimulator
from aapl_pipeline.performance_metric import (
    MetricContext,
    MetricRegistry,
    default_registry,
)
from aapl_pipeline.rolling_model import RollingClassifier, ScoreFunction
from aapl_pipeline.weekly_builder import WeeklyDataset


class StrategyPipeline:
    """Top level entry point.

    Almost every component is replaceable. The constructor accepts the
    full set of plug ins and falls back to sensible defaults when one is
    omitted.
    """

    def __init__(
        self,
        config: PipelineConfig,
        feature_calculators: Sequence[FeatureCalculator],
        modeller: Optional[Modeller] = None,
        baselines: Optional[Sequence[Baseline]] = None,
        registry: Optional[MetricRegistry] = None,
        loader: Optional[DataLoader] = None,
    ) -> None:
        self.config = config
        self.feature_calculators = list(feature_calculators)
        self.loader = loader or DataLoader(path=config.data_path)
        self.registry = registry or default_registry()
        self.baselines = list(baselines) if baselines is not None else self._default_baselines()
        self.modeller = modeller or self._default_modeller()
        self.rng = config.make_rng()

    def _default_modeller(self) -> Modeller:
        c = self.config
        return DecisionTreeModeller(
            depth_grid=c.depth_grid,
            leaf_grid=c.leaf_grid,
            fixed_params=c.fixed_tree_params,
        )

    def _default_baselines(self) -> List[Baseline]:
        c = self.config
        return [
            AlwaysTradeBaseline(c.initial_bank, c.upper_thresh, c.lower_thresh),
            RandomTraderBaseline(c.initial_bank, c.upper_thresh, c.lower_thresh),
            AlternateWeekBaseline(c.initial_bank, c.upper_thresh, c.lower_thresh),
            WeightedCoinBaseline(c.initial_bank, c.upper_thresh, c.lower_thresh),
        ]

    # Pipeline phases.

    def _load_and_build_weekly(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        daily = self.loader.load(cutoff_date=self.config.cutoff_date)
        builder = WeeklyDataset(week_period_freq=self.config.week_period_freq)
        weekly = builder.build(daily)
        return builder.df, weekly

    def _compute_features(
        self, daily_df: pd.DataFrame, weekly_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, List[str]]:
        feature_frames = [calc.compute(daily_df, weekly_df) for calc in self.feature_calculators]
        feature_columns: List[str] = []
        for calc in self.feature_calculators:
            feature_columns.extend(calc.feature_names)

        all_features = (
            pd.concat(feature_frames, axis=1)
            if feature_frames
            else pd.DataFrame(index=weekly_df.index)
        )
        weekly_full = weekly_df.join(all_features, how="left")

        required_cols = feature_columns + ["week_type", "thu/tue"]
        weekly_full = weekly_full.dropna(subset=required_cols)
        return weekly_full, feature_columns

    def _build_subsets(self, df_final: pd.DataFrame) -> List[tuple[int, pd.DataFrame]]:
        """Linspace subset starts after subset_start_date, plus a final
        n_weeks window that ends the simulation. Mirrors the SR notebook.
        """
        c = self.config

        if c.subset_start_date is not None:
            ss_dt = pd.to_datetime(c.subset_start_date)
            mask = df_final["week_start_date"] >= ss_dt
            base_start = int(mask.idxmax()) if mask.any() else len(df_final)
        else:
            base_start = 0

        last_start = len(df_final) - c.n_weeks
        if last_start < base_start:
            raise ValueError("Not enough data to produce even the final window.")

        if c.num_subsets < 0:
            raise ValueError("num_subsets must be >= 0")

        if c.num_subsets > 0:
            offsets = np.linspace(0, last_start - base_start, c.num_subsets + 2, dtype=int)
            start_points = offsets[0:c.num_subsets] + base_start
        else:
            start_points = np.array([], dtype=int)

        subsets: List[tuple[int, pd.DataFrame]] = []
        for s in start_points:
            subsets.append((int(s), df_final.iloc[int(s):int(s) + c.n_weeks]))
        subsets.append((int(last_start), df_final.iloc[int(last_start):int(last_start) + c.n_weeks]))
        return subsets

    def _historical_mc(
        self,
        df_final: pd.DataFrame,
        subsets: List[tuple[int, pd.DataFrame]],
        mc: MonteCarloSimulator,
    ) -> tuple[np.ndarray, np.ndarray, List[float]]:
        """Run the historical block per subset and concatenate results.

        Skips the first subset (no history to sample from) and uses the
        history strictly before each subset's start as the empirical
        source for TP and FP multipliers.
        """
        all_sims: List[np.ndarray] = []
        actual_balances: List[float] = []
        null_percentiles: List[float] = []

        for i, (start, sub) in enumerate(subsets):
            if i == 0:
                continue

            history = df_final.iloc[:start]
            tp_hist = history.loc[history["Outcome"] == "TP", "thu_tue"].values
            fp_hist = history.loc[history["Outcome"] == "FP", "thu_tue"].values

            # Need at least two of each so the empirical CDF is meaningful.
            if len(tp_hist) < 2 or len(fp_hist) < 2:
                continue

            tp_sampler = EmpiricalSampler(tp_hist, self.rng)
            fp_sampler = EmpiricalSampler(fp_hist, self.rng)
            p_hist = MonteCarloSimulator.outcome_probs_from(history)

            sims = mc.run(p_hist, tp_sampler, fp_sampler)
            all_sims.append(sims)

            actual = mc.run_actual(sub)
            actual_balances.append(actual)
            null_percentiles.append(float(np.mean(sims <= actual)))

        sim_all = (
            np.concatenate(all_sims)
            if all_sims
            else np.array([self.config.initial_bank])
        )
        actual_arr = (
            np.array(actual_balances)
            if actual_balances
            else np.array([self.config.initial_bank])
        )
        if not null_percentiles:
            null_percentiles = [0.5]
        return sim_all, actual_arr, null_percentiles

    def _future_mc(self, df_final: pd.DataFrame, mc: MonteCarloSimulator) -> np.ndarray:
        """Future block samples from the entire df_final."""
        tp_all = df_final.loc[df_final["Outcome"] == "TP", "thu_tue"].values
        fp_all = df_final.loc[df_final["Outcome"] == "FP", "thu_tue"].values
        tp_sampler = EmpiricalSampler(tp_all, self.rng)
        fp_sampler = EmpiricalSampler(fp_all, self.rng)
        p_all = MonteCarloSimulator.outcome_probs_from(df_final)
        return mc.run(p_all, tp_sampler, fp_sampler)

    def _baseline_ratios(
        self,
        subsets: List[tuple[int, pd.DataFrame]],
        mc: MonteCarloSimulator,
    ) -> Dict[str, float]:
        per_baseline: Dict[str, List[float]] = {b.name: [] for b in self.baselines}

        for _, sub in subsets:
            model_bal = mc.run_actual(sub)
            for b in self.baselines:
                base_bal = b.run(sub, self.rng)
                if base_bal == 0:
                    per_baseline[b.name].append(float("nan"))
                else:
                    per_baseline[b.name].append(model_bal / base_bal)

        return {
            name: float(np.nanmean(vals)) if vals else float("nan")
            for name, vals in per_baseline.items()
        }

    # Public entry point.

    def run(self, verbose: bool = True) -> Dict[str, Any]:
        c = self.config

        daily, weekly = self._load_and_build_weekly()
        weekly_full, feature_columns = self._compute_features(daily, weekly)

        score_fn = ScoreFunction(c.alpha_p, c.alpha_c, c.p_min, c.c_min)
        clf = RollingClassifier(
            modeller=self.modeller,
            valid_weeks=c.valid_weeks,
            thresholds_tested=c.thresholds_tested,
            score_function=score_fn,
        )
        df_final = clf.run(weekly_full, feature_columns, target_column="week_type", progress=verbose)

        mc = MonteCarloSimulator(
            n_trajectories=c.n_trajectories,
            n_weeks=c.n_weeks,
            initial_bank=c.initial_bank,
            upper_thresh=c.upper_thresh,
            lower_thresh=c.lower_thresh,
            rng=self.rng,
        )

        subsets = self._build_subsets(df_final)
        sim_all, actual_arr, null_pcts = self._historical_mc(df_final, subsets, mc)
        future_sims = self._future_mc(df_final, mc)
        baseline_ratios = self._baseline_ratios(subsets, mc)

        ctx = MetricContext(
            df_final=df_final,
            confusion=clf.confusion,
            initial_bank=c.initial_bank,
            sim_all=sim_all,
            actual_balances=actual_arr,
            null_percentiles=null_pcts,
            future_sims=future_sims,
            upper_thresh=c.upper_thresh,
            lower_thresh=c.lower_thresh,
            uniformity_binsize=c.uniformity_binsize,
            baseline_ratios=baseline_ratios,
        )

        report = self.registry.run_all(ctx)
        if verbose:
            pprint(report)

        # Stash artifacts for the caller.
        self.last_df_final = df_final
        self.last_weekly_full = weekly_full
        self.last_report = report
        self.last_future_sims = future_sims
        self.last_sim_all = sim_all
        self.last_baseline_ratios = baseline_ratios

        return report


# Factory helpers. Both pipelines share the same backtest layout. They
# differ only in the feature calculator and the parameters that drive it.

def make_tue_thu_pipeline(config: Optional[PipelineConfig] = None) -> StrategyPipeline:
    """Pipeline with the three normalized opens as features.

    Same backtest layout as the SR pipeline. Pass a partially filled
    config to override any defaults.
    """
    config = config or PipelineConfig()
    return StrategyPipeline(
        config=config,
        feature_calculators=[TueThuNormalizedFeatures()],
    )


def make_support_resistance_pipeline(config: Optional[PipelineConfig] = None) -> StrategyPipeline:
    """Pipeline with the nine envelope features.

    Reads SR specific parameters off the config. Pass a partially filled
    config to override any defaults.
    """
    config = config or PipelineConfig()
    sr = SupportResistanceFeatures(
        smooth_vals=config.sr_smooth_vals,
        window_vals=config.sr_window_vals,
        envelope_a=config.envelope_a,
        envelope_b=config.envelope_b,
        max_iter=config.sr_max_iter,
        tol=config.sr_tol,
    )
    return StrategyPipeline(
        config=config,
        feature_calculators=[sr],
    )
