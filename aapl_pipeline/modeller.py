"""Modeller hierarchy.

Same idea as FeatureCalculator: keep the orchestration code generic, push
all the model specific bits into subclasses. The rolling backtest only
talks to two methods, `make_estimator` and `param_grid`, so adding a new
model is a one class drop in.

Standalone users can also use a Modeller directly. After `.fit(X, y)` the
instance holds the trained estimator and `.predict_proba_positive(X)` and
`.predict(X, threshold)` are ready to call.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import product
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


class Modeller(ABC):
    """OO wrapper around a binary classifier.

    Subclasses define how to build an estimator and which hyperparameters
    to search. The base class handles fit, predict and probability
    extraction so every subclass behaves identically from the outside.
    """

    def __init__(self) -> None:
        self._estimator: Any = None
        self._last_hyperparams: Dict[str, Any] = {}

    @abstractmethod
    def make_estimator(self, **hyperparams: Any) -> Any:
        """Return a fresh, unfitted estimator with the given hyperparameters."""

    @abstractmethod
    def param_grid(self) -> Iterable[Dict[str, Any]]:
        """Yield hyperparameter dicts to evaluate during the rolling search."""

    @property
    def estimator(self) -> Any:
        if self._estimator is None:
            raise RuntimeError("Modeller has no fitted estimator yet. Call fit() first.")
        return self._estimator

    @property
    def last_hyperparams(self) -> Dict[str, Any]:
        return dict(self._last_hyperparams)

    def fit(self, X: pd.DataFrame, y: pd.Series, **hyperparams: Any) -> "Modeller":
        """Fit a fresh estimator with the given hyperparameters on (X, y)."""
        self._estimator = self.make_estimator(**hyperparams)
        self._estimator.fit(X, y)
        self._last_hyperparams = dict(hyperparams)
        return self

    def predict_proba_positive(self, X: pd.DataFrame) -> np.ndarray:
        """Probability of the positive class for each row in X."""
        return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Hard predictions using the supplied probability threshold."""
        return (self.predict_proba_positive(X) > threshold).astype(int)


class DecisionTreeModeller(Modeller):
    """sklearn DecisionTreeClassifier with a (max_depth, min_samples_leaf) grid.

    Reproduces the original behaviour of the support resistance and
    final_function_tue_thu pipelines.
    """

    def __init__(
        self,
        depth_grid: Sequence[int],
        leaf_grid: Sequence[int],
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.depth_grid = list(depth_grid)
        self.leaf_grid = list(leaf_grid)
        self.fixed_params = dict(fixed_params or {})

    def make_estimator(
        self,
        max_depth: int,
        min_samples_leaf: int,
        **extra: Any,
    ) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            **{**self.fixed_params, **extra},
        )

    def param_grid(self) -> Iterable[Dict[str, Any]]:
        for d, leaf in product(self.depth_grid, self.leaf_grid):
            yield {"max_depth": d, "min_samples_leaf": leaf}


class RandomForestModeller(Modeller):
    """sklearn RandomForestClassifier with an (n_estimators, max_depth) grid.

    Provided as a worked example of how to add a different model. Drop an
    instance of this class into StrategyPipeline(modeller=...) and the
    rolling backtest swaps over without any other changes.
    """

    def __init__(
        self,
        n_estimators_grid: Sequence[int] = (100, 300),
        depth_grid: Sequence[Optional[int]] = (3, 6, None),
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.n_estimators_grid = list(n_estimators_grid)
        self.depth_grid = list(depth_grid)
        self.fixed_params = dict(fixed_params or {
            "random_state": 42,
            "class_weight": "balanced",
            "n_jobs": -1,
        })

    def make_estimator(
        self,
        n_estimators: int,
        max_depth: Optional[int],
        **extra: Any,
    ) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            **{**self.fixed_params, **extra},
        )

    def param_grid(self) -> Iterable[Dict[str, Any]]:
        for n, d in product(self.n_estimators_grid, self.depth_grid):
            yield {"n_estimators": n, "max_depth": d}
