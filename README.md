# AAPL Weekly Trading Pipeline

A modular research pipeline for evaluating a Tuesday open to Thursday
open trading strategy on Apple daily price data. Every component
(feature family, model, baseline trader, performance metric, report
comparator) is a class. Adding a new component is a matter of writing
one subclass. The orchestrator does not change.

## Table of Contents

- [[#Quick Start]]
- [[#Repository Layout]]
- [[#Data Flow]]
- [[#Class Hierarchies]]
    - [[#FeatureCalculator]]
    - [[#Modeller]]
    - [[#Baseline]]
    - [[#PerformanceMetric]]
    - [[#ReportComparator]]
- [[#Configuration]]
- [[#File Reference]]
- [[#Extending the Pipeline]]
- [[#Inspecting Results]]
- [[#Comparing Two Models]]
    - [[#Where to Change Weights]]

## Quick Start

Install the package in editable mode from the project root, then import
from anywhere.

```bash
pip install -e .
```

```python
from aapl_pipeline import PipelineConfig, make_support_resistance_pipeline

params = {
    "VALID_WEEKS": 52,
    "depth_grid": [2, 3, 4, 6],
    "leaf_grid": [2, 3, 4, 6],
    "FIXED": {
        "criterion": "entropy",
        "min_samples_split": 6,
        "class_weight": "balanced",
        "random_state": 42,
    },
    "alpha_p": 1.0, "alpha_c": 0.01, "p_min": 0.55, "c_min": 0.10,
    "n_trajectories": 100000, "n_weeks": 100,
    "initial_bank": 100.0, "upper_thresh": 200.0, "lower_thresh": 60.0,
    "rng_seed": 42,
    "cutoff_date": "1990-01-01",
    "subset_start_date": "2000-01-01",
    "num_subsets": 5,
    "ENVELOPE_A": 1.0, "ENVELOPE_B": 100.0,
    "SR_SMOOTH_VALS": [2, 4, 12], "SR_WINDOW_VALS": [4, 26, 52],
}

config = PipelineConfig.from_params_dict(params)
pipe = make_support_resistance_pipeline(config)
report = pipe.run()
```

The report dictionary contains every diagnostic from the run, including
the raw Monte Carlo arrays under `report["historical_mc"]["simulations"]`
and `report["future_mc"]["simulations"]`.

## Repository Layout

```
Research-/
  aapl_pipeline/
    __init__.py
    config.py
    data_loader.py
    weekly_builder.py
    feature_calculator.py
    modeller.py
    rolling_model.py
    monte_carlo.py
    baselines.py
    performance_metric.py
    comparator.py
    pipeline.py
  notebooks/
    exploration.ipynb
    run_pipeline.ipynb
    final_function_tue_thu.ipynb
    suppor_and_resistance.ipynb
  README.md
  pyproject.toml
```

## Data Flow

| Stage | Component | Responsibility |
|-------|-----------|----------------|
| 1 | `DataLoader` | Read the cleaned daily CSV |
| 2 | `WeeklyDataset` | Aggregate to a Tue Thu weekly skeleton |
| 3 | `FeatureCalculator` | Compute one or more feature families |
| 4 | `RollingClassifier` | Walk forward week by week with grid plus threshold search |
| 5 | `MonteCarloSimulator` | Simulate bank trajectories |
| 6 | `Baseline` | Replay naive trader policies |
| 7 | `MetricRegistry` | Aggregate every diagnostic into one report dict |
| 8 | `ReportComparator` | (Optional) compare two report dicts and pick a winner |

`StrategyPipeline` wires stages 1 through 7 together. The comparator is
called separately once you have two reports.

## Class Hierarchies

The package is organised around five parent classes. Each parent is
abstract and exists to make new variants easy to plug in.

### FeatureCalculator

Defined in `aapl_pipeline/feature_calculator.py`. Subclasses declare
which feature columns they emit and how to compute them given the
daily and weekly frames.

| Subclass | What it emits | Notes |
|----------|---------------|-------|
| `TueThuNormalizedFeatures` | `Norm_PrevThu_Open`, `Norm_PrevFri_Open`, `Norm_Tue_Open` | Causal z scores. Expanding mean and std are shifted by one day to avoid look ahead. |
| `SupportResistanceFeatures` | Nine columns `SR_{smooth}_{window}` | Two iteratively reweighted line fits per window produce support and resistance lines. The feature is the position of Tuesday open inside the envelope. |

> [!note]
> Adding a new family of features means writing one new subclass and
> dropping an instance into `feature_calculators=[...]`. The pipeline
> concatenates outputs of every registered calculator.

### Modeller

Defined in `aapl_pipeline/modeller.py`. Wraps a binary classifier so
the rolling backtest can stay generic. Subclasses define
`make_estimator(**hp)` and `param_grid()`. The base class implements
`fit(X, y, **hp)`, `predict_proba_positive(X)`, and
`predict(X, threshold)`.

| Subclass | Estimator | Hyperparameter grid |
|----------|-----------|---------------------|
| `DecisionTreeModeller` | `sklearn.tree.DecisionTreeClassifier` | `(max_depth, min_samples_leaf)` |
| `RandomForestModeller` | `sklearn.ensemble.RandomForestClassifier` | `(n_estimators, max_depth)` |

> [!info]
> Both original notebooks (Tuesday Thursday and Support Resistance)
> use `DecisionTreeModeller`. `RandomForestModeller` is provided as a
> worked example of how a different model is plugged in.

### Baseline

Defined in `aapl_pipeline/baselines.py`. Replays a fixed naive policy
over a slice of the per week prediction frame and returns the final
bank. The pipeline computes the ratio of the model bank to each
baseline bank and averages across the historical subsets.

| Subclass | Policy |
|----------|--------|
| `AlwaysTradeBaseline` | Trade every week |
| `RandomTraderBaseline` | Coin matched to model chattiness |
| `AlternateWeekBaseline` | Trade every other week |
| `WeightedCoinBaseline` | Coin matched to historical good week rate |

### PerformanceMetric

Defined in `aapl_pipeline/performance_metric.py`. Every value in the
report card is a subclass that returns one number from `compute(ctx)`
and declares three things on the class:

- `group` the report dict key it belongs to
- `name` the entry name within that group
- `direction` one of `"maximize"`, `"minimize"`, or `"neutral"`

A `MetricRegistry` holds metric instances and runs all of them in one
pass to produce the nested report dictionary. `default_registry()`
returns a registry preloaded with every metric below.

#### Internal metrics (confusion based)

| Class | Direction | Meaning |
|-------|-----------|---------|
| `PrecisionOverall` | maximize | TP / (TP + FP) |
| `ChattinessOverall` | neutral | (TP + FP) / (TP + FN) |
| `CorrectnessRate` | maximize | (TP + TN) / total |
| `TradeFrequency` | neutral | Fraction of weeks the model traded |
| `MistakeAsymmetry` | maximize | Average TP gain plus average FP loss, in percent |
| `LongestStreak("TP")` | neutral | Longest run of TP outcomes |
| `LongestStreak("FP")` | minimize | Longest run of FP outcomes |
| `FPRateWhenPredictedPositive` | minimize | FP / (TP + FP) |

#### Historical Monte Carlo

| Class | Direction | Meaning |
|-------|-----------|---------|
| `SimulatedMean` | maximize | Mean simulated bank across all historical subsets |
| `SimulatedMedian` | maximize | Median simulated bank |
| `KSDistance` | neutral | KS distance between actual and simulated bank distributions |
| `KSPValue` | neutral | KS p value |
| `AverageNullPercentile` | maximize | Mean rank of actual balance inside its simulated null |
| `HistoricalSimulations` | neutral | Concatenated bank trajectories array |
| `HistoricalActuals` | neutral | Actual bank balance per subset |
| `HistoricalNullPercentiles` | neutral | Per subset null percentile rank |

#### Future Monte Carlo

| Class | Direction | Meaning |
|-------|-----------|---------|
| `FutureMean` | maximize | Mean forward looking bank |
| `FutureMedian` | maximize | Median forward looking bank |
| `ProbAboveInitial` | maximize | P(final bank > initial bank) |
| `ProbSuccess` | maximize | P(final bank >= upper_thresh) |
| `ProbFailure` | minimize | P(final bank <= lower_thresh) |
| `ProbUncertain` | neutral | P(final bank in the do nothing band) |
| `FutureSimulations` | neutral | Forward looking trajectories array |

#### Statistical tests

Each test is split into individual scalar metrics so every report
entry is one number.

| Class | Direction | Meaning |
|-------|-----------|---------|
| `RunsTestZ` | neutral | Wald Wolfowitz runs test z statistic |
| `RunsTestPValue` | neutral | Wald Wolfowitz runs test p value |
| `ChiSquareStatistic` | neutral | Chi square statistic for time uniformity |
| `ChiSquarePValue` | neutral | Chi square p value |
| `ChiSquareDoF` | neutral | Degrees of freedom |

#### Baseline ratios

| Class | Direction | Meaning |
|-------|-----------|---------|
| `BaselineRatio` | maximize | Model bank divided by the named baseline bank, averaged across subsets |

> [!tip]
> `BaselineRatio` is registered once per baseline, so the report dict
> has one entry per baseline under `baseline_comparison`. They all
> share `direction = "maximize"`.

### ReportComparator

Defined in `aapl_pipeline/comparator.py`. Takes two report dicts and
decides which model is better. The parent class is abstract; subclasses
implement the `compare(...)` method using whatever rule they want.

| Subclass | Rule |
|----------|------|
| `WeightedVoteComparator` | Each metric casts `weight` votes for whichever side wins under its `direction`. Verdict is the higher weighted total. |

`WeightedVoteComparator` walks every metric in the registry, looks up
both reports, decides the winner per metric, and adds `weight` to that
side's running total. Default weight is 1.0 for any metric whose
direction is `"maximize"` or `"minimize"`, and 0.0 for anything
`"neutral"`. Override per metric via the `weights={"group.name": w}`
dict.

The result object is a `ComparisonResult` with three handy methods:

- `result.voting_rows()` dataframe of metrics that contributed
- `result.to_dataframe()` full breakdown including neutral rows
- `repr(result)` quick text summary

> [!info]
> Statistical tests show up in the table for context but do not vote
> by default because their direction is neutral. To make them vote,
> add a non zero weight: `weights={"uniformity_test.p_value": 1.0}`.

## Configuration

`PipelineConfig` is a dataclass that holds every parameter the pipeline
might consume. Defaults match the suppor_and_resistance setup, which
is the canonical backtest layout the project uses.

| Group | Fields |
|-------|--------|
| Data | `data_path`, `cutoff_date`, `week_period_freq` |
| Rolling CV | `valid_weeks`, `depth_grid`, `leaf_grid`, `thresholds_tested`, `fixed_tree_params` |
| Score function | `alpha_p`, `alpha_c`, `p_min`, `c_min` |
| Monte Carlo | `n_trajectories`, `n_weeks`, `initial_bank`, `upper_thresh`, `lower_thresh`, `rng_seed` |
| Subsets | `num_subsets`, `subset_start_date` |
| Statistical tests | `uniformity_binsize` |
| Support resistance | `envelope_a`, `envelope_b`, `sr_max_iter`, `sr_tol`, `sr_smooth_vals`, `sr_window_vals` |

`PipelineConfig.from_params_dict(d)` accepts the original notebook
style flat dict. It maps the legacy uppercase keys (`VALID_WEEKS`,
`FIXED`, `ENVELOPE_A`, `SR_SMOOTH_VALS`) to the dataclass fields and
treats `n_subsets` and `num_subsets` as aliases. Unknown keys (notably
`df`) are ignored, so the legacy dicts pass through unchanged.

## File Reference

### Package files

| File | Holds |
|------|-------|
| `aapl_pipeline/__init__.py` | Re export of the public surface |
| `aapl_pipeline/config.py` | `PipelineConfig` dataclass and `from_params_dict` |
| `aapl_pipeline/data_loader.py` | `DataLoader` for reading the cleaned daily CSV |
| `aapl_pipeline/weekly_builder.py` | `WeeklyDataset` per week table builder |
| `aapl_pipeline/feature_calculator.py` | `FeatureCalculator` parent and the two feature subclasses |
| `aapl_pipeline/modeller.py` | `Modeller` parent and the two model subclasses |
| `aapl_pipeline/rolling_model.py` | `RollingClassifier` walk forward backtest and `ScoreFunction` |
| `aapl_pipeline/monte_carlo.py` | `EmpiricalSampler` and `MonteCarloSimulator` |
| `aapl_pipeline/baselines.py` | `Baseline` parent and the four trader subclasses |
| `aapl_pipeline/performance_metric.py` | `PerformanceMetric` parent, every metric subclass, `MetricRegistry`, `default_registry()` |
| `aapl_pipeline/comparator.py` | `ReportComparator` parent and `WeightedVoteComparator` subclass |
| `aapl_pipeline/pipeline.py` | `StrategyPipeline` orchestrator and the two factory helpers |

### Notebooks

All notebooks live under `notebooks/` and import from the installed
package.

| File | Purpose |
|------|---------|
| `notebooks/exploration.ipynb` | Consolidated EDA log of what we learned about the data before the pipeline existed |
| `notebooks/run_pipeline.ipynb` | Thin runner notebook on top of the package, shows both factory helpers in action |
| `notebooks/final_function_tue_thu.ipynb` | Self contained Tuesday Thursday pipeline run with the SR backtest layout |
| `notebooks/suppor_and_resistance.ipynb` | Self contained Support Resistance pipeline run, the canonical reference for the backtest layout |

### Project files

| File | Purpose |
|------|---------|
| `README.md` | This document |
| `pyproject.toml` | Package metadata, dependencies, and build configuration. Run `pip install -e .` from this directory to make the package importable from anywhere |

## Extending the Pipeline

Adding a new component follows the same recipe regardless of which
parent class it belongs to: write one subclass, drop the instance into
the pipeline. Nothing inside the orchestrator changes.

### A new feature family

```python
from aapl_pipeline import FeatureCalculator, StrategyPipeline, PipelineConfig

class VolumeRatioFeatures(FeatureCalculator):
    @property
    def feature_names(self):
        return ["vol_ratio_4w"]

    def compute(self, daily_df, weekly_df):
        # return a dataframe indexed by weekly_df.index
        ...

pipe = StrategyPipeline(
    config=PipelineConfig(),
    feature_calculators=[VolumeRatioFeatures()],
)
```

### A new model

```python
from aapl_pipeline import Modeller
from xgboost import XGBClassifier

class XGBModeller(Modeller):
    def make_estimator(self, max_depth, learning_rate, **extra):
        return XGBClassifier(
            max_depth=max_depth, learning_rate=learning_rate, **extra
        )

    def param_grid(self):
        for d in (3, 5, 7):
            for lr in (0.05, 0.1):
                yield {"max_depth": d, "learning_rate": lr}
```

Pass it via `modeller=XGBModeller()` when constructing the pipeline.

### A new metric

```python
from aapl_pipeline import PerformanceMetric, default_registry

class SortinoRatio(PerformanceMetric):
    group = "internal_metrics"
    name = "sortino_ratio"
    direction = "maximize"

    def compute(self, ctx):
        ...

reg = default_registry()
reg.register(SortinoRatio())
```

Pass it via `registry=reg` when constructing the pipeline.

### A new baseline

```python
from aapl_pipeline import Baseline

class BuyAndHoldBaseline(Baseline):
    @property
    def name(self):
        return "vs_buy_and_hold"

    def run(self, sub, rng):
        ...
```

Pass it via `baselines=[..., BuyAndHoldBaseline(...)]` when constructing
the pipeline.

### A new comparator

```python
from aapl_pipeline import ReportComparator, ComparisonResult

class RelativeMagnitudeComparator(ReportComparator):
    def compare(self, report_a, report_b, name_a="A", name_b="B"):
        # walk self.registry, score each metric on its magnitude
        ...
        return ComparisonResult(...)
```

## Inspecting Results

```python
report = pipe.run()

# Scalar metrics
print(report["internal_metrics"]["precision_overall"])
print(report["future_mc"]["prob_success"])

# Raw Monte Carlo arrays
import numpy as np
print(np.quantile(report["future_mc"]["simulations"], [0.25, 0.5, 0.75]))

# Per week predictions
df_final = pipe.last_df_final
df_final.head()

# Saved on the pipeline instance for convenience
pipe.last_future_sims
pipe.last_baseline_ratios
```

## Comparing Two Models

```python
from aapl_pipeline import (
    PipelineConfig, default_registry,
    make_tue_thu_pipeline, make_support_resistance_pipeline,
    WeightedVoteComparator,
)

config = PipelineConfig()

pipe_tt = make_tue_thu_pipeline(config)
pipe_sr = make_support_resistance_pipeline(config)

report_tt = pipe_tt.run()
report_sr = pipe_sr.run()

registry = default_registry()
comparator = WeightedVoteComparator(registry)
result = comparator.compare(report_sr, report_tt, name_a="SR", name_b="TueThu")

print(result)              # verdict, raw counts, weighted totals
result.voting_rows()       # dataframe of metrics that contributed
result.to_dataframe()      # full breakdown including neutral rows
```

`result.verdict` is the name of the winner (`"SR"`, `"TueThu"`, or
`"tie"`). `result.weighted_total` is a dict like
`{"SR": 6.0, "TueThu": 14.0}`. `result.raw_count` is a dict like
`{"SR": 4, "TueThu": 12, "ties": 1}`.

When raw count and weighted total disagree, you have learned something:
the models trade off differently and the verdict is being driven by
your weights.

### Where to Change Weights

Weights live on the `WeightedVoteComparator` instance. There are three
places you can set them.

**1. At construction.** Pass a `weights` dict keyed by `"group.name"`.

```python
comparator = WeightedVoteComparator(
    registry,
    weights={
        "baseline_comparison.vs_always_trade":      5.0,
        "future_mc.prob_failure":                   5.0,
        "future_mc.prob_success":                   3.0,
        "internal_metrics.precision_overall":       3.0,
    },
)
```

**2. After construction.** `comparator.weights` is a regular dict, so
mutate it directly.

```python
comparator.weights["baseline_comparison.vs_random_trader"] = 2.0
comparator.weights.pop("internal_metrics.precision_overall", None)
```

**3. Per call.** Build a fresh comparator with different weights for
the same registry. Cheap, since the registry is reused.

```python
heavier_baseline = WeightedVoteComparator(
    registry,
    weights={
        "baseline_comparison.vs_always_trade":      4.0,
        "baseline_comparison.vs_random_trader":     4.0,
        "baseline_comparison.vs_alternate_trader":  4.0,
        "baseline_comparison.vs_weighted_coin":     4.0,
    },
)
result = heavier_baseline.compare(report_sr, report_tt, name_a="SR", name_b="TueThu")
```

The key for each entry is `"<group>.<name>"`, exactly the path used
inside the report dict. To find the right key for a metric, look it up
in the [[#PerformanceMetric]] tables above. Every group plus name pair
is the same string you would put in `weights`.

> [!note] Default weights
> Any metric not listed in `weights` keeps its default. Metrics with
> `direction = "maximize"` or `"minimize"` default to weight 1.0.
> Metrics with `direction = "neutral"` default to 0.0, so raw arrays
> and statistical tests do not vote unless you explicitly weight them.

> [!warning] Negative weights
> The current implementation accepts any float, including negatives.
> A negative weight inverts which side that metric supports, which is
> almost always not what you want. Stick to non negative weights
> unless you have a deliberate reason.
