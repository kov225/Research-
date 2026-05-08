"""AAPL weekly trading pipeline.

Public surface kept small on purpose. Most notebooks only need
PipelineConfig, StrategyPipeline, and one of the make_*_pipeline
factory helpers. The class hierarchies (FeatureCalculator, Modeller,
PerformanceMetric) are exposed so new subclasses can be plugged in
without touching the orchestrator.
"""

from aapl_pipeline.config import PipelineConfig
from aapl_pipeline.data_loader import DataLoader
from aapl_pipeline.weekly_builder import WeeklyDataset
from aapl_pipeline.feature_calculator import (
    FeatureCalculator,
    TueThuNormalizedFeatures,
    SupportResistanceFeatures,
)
from aapl_pipeline.modeller import (
    Modeller,
    DecisionTreeModeller,
    RandomForestModeller,
)
from aapl_pipeline.rolling_model import RollingClassifier, ScoreFunction
from aapl_pipeline.monte_carlo import EmpiricalSampler, MonteCarloSimulator
from aapl_pipeline.baselines import (
    Baseline,
    AlwaysTradeBaseline,
    RandomTraderBaseline,
    AlternateWeekBaseline,
    WeightedCoinBaseline,
)
from aapl_pipeline.performance_metric import (
    PerformanceMetric,
    MetricContext,
    MetricRegistry,
    default_registry,
)
from aapl_pipeline.comparator import (
    ReportComparator,
    WeightedVoteComparator,
    ComparisonResult,
)
from aapl_pipeline.pipeline import (
    StrategyPipeline,
    make_tue_thu_pipeline,
    make_support_resistance_pipeline,
)

__all__ = [
    "PipelineConfig",
    "DataLoader",
    "WeeklyDataset",
    "FeatureCalculator",
    "TueThuNormalizedFeatures",
    "SupportResistanceFeatures",
    "Modeller",
    "DecisionTreeModeller",
    "RandomForestModeller",
    "RollingClassifier",
    "ScoreFunction",
    "EmpiricalSampler",
    "MonteCarloSimulator",
    "Baseline",
    "AlwaysTradeBaseline",
    "RandomTraderBaseline",
    "AlternateWeekBaseline",
    "WeightedCoinBaseline",
    "PerformanceMetric",
    "MetricContext",
    "MetricRegistry",
    "default_registry",
    "ReportComparator",
    "WeightedVoteComparator",
    "ComparisonResult",
    "StrategyPipeline",
    "make_tue_thu_pipeline",
    "make_support_resistance_pipeline",
]
