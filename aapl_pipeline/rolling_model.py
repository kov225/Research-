"""Rolling backtest with grid search over a Modeller and threshold.

The rolling logic itself does not care which model is being used. It
only knows the Modeller interface: ask for hyperparameter combinations,
fit a fresh estimator on each, score on a validation window, pick the
best one, predict the next week. Swap a Modeller subclass to change the
underlying model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from tqdm import tqdm

from aapl_pipeline.modeller import Modeller


@dataclass
class ScoreFunction:
    """Custom validation score that mixes precision and chattiness.

    The exponential form lets the user reward precision strongly while
    only nudging chattiness. Numerically the same shape as both source
    notebooks.
    """
    alpha_p: float = 1.0
    alpha_c: float = 0.01
    p_min: float = 0.55
    c_min: float = 0.10

    def __call__(self, tp: int, fp: int, fn: int) -> float:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        chat = (tp + fp) / (tp + fn) if (tp + fn) > 0 else 0.0
        s = np.exp(self.alpha_p * (prec - self.p_min) + self.alpha_c * (chat - self.c_min))
        if np.isnan(s) or np.isinf(s):
            return 0.0
        return float(s)


class RollingClassifier:
    """Walks forward week by week.

    For every week we train on the past, validate on the most recent
    valid_weeks window, search the modeller's hyperparameter grid and the
    threshold list, then predict one step ahead with the best combination.
    The output is a per week dataframe of predictions and outcomes which
    feeds every downstream metric and the Monte Carlo block.
    """

    def __init__(
        self,
        modeller: Modeller,
        valid_weeks: int,
        thresholds_tested: Sequence[float],
        score_function: ScoreFunction,
    ) -> None:
        self.modeller = modeller
        self.valid_weeks = valid_weeks
        self.thresholds_tested = list(thresholds_tested)
        self.score_function = score_function

    def run(
        self,
        weekly_full: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str = "week_type",
        progress: bool = True,
    ) -> pd.DataFrame:
        """Execute the rolling backtest.

        weekly_full must already have features joined and rows with NaNs
        dropped. Returns a per week dataframe with True_Label, Pred_Label,
        Outcome, thu_tue, and the chosen hyperparameters.
        """
        TP = TN = FP = FN = 0
        rows = []
        n = len(weekly_full)

        iterator = range(self.valid_weeks + 1, n)
        if progress:
            iterator = tqdm(iterator, desc="Rolling simulation")

        for t in iterator:
            val_start = max(0, t - self.valid_weeks)
            training = weekly_full.iloc[:val_start]
            validation = weekly_full.iloc[val_start:t]
            test = weekly_full.iloc[[t]]

            if len(training[target_column].unique()) < 2:
                continue

            train_X = training[list(feature_columns)]
            train_y = training[target_column]
            val_X = validation[list(feature_columns)]
            val_y = validation[target_column]
            test_X = test[list(feature_columns)]
            test_y = test[target_column]

            best_score = -np.inf
            best_estimator = None
            best_hp = None
            best_thr = None

            for hp in self.modeller.param_grid():
                # Build a fresh estimator per candidate so the inner loop
                # cannot pollute later iterations through shared state.
                candidate = self.modeller.make_estimator(**hp)
                candidate.fit(train_X, train_y)
                probs_val = candidate.predict_proba(val_X)[:, 1]

                for thr in self.thresholds_tested:
                    preds_val = (probs_val > thr).astype(int)
                    tp = int(((preds_val == 1) & (val_y == 1)).sum())
                    fp = int(((preds_val == 1) & (val_y == 0)).sum())
                    fn = int(((preds_val == 0) & (val_y == 1)).sum())
                    sc = self.score_function(tp, fp, fn)
                    if sc > best_score:
                        best_score = sc
                        best_estimator = candidate
                        best_hp = hp
                        best_thr = thr

            if best_estimator is None:
                continue

            p_hat = best_estimator.predict_proba(test_X)[0, 1]
            pred = int(p_hat > best_thr)
            true = int(test_y.iloc[0])

            if pred == 1 and true == 1:
                TP += 1
                outcome = "TP"
            elif pred == 0 and true == 0:
                TN += 1
                outcome = "TN"
            elif pred == 1 and true == 0:
                FP += 1
                outcome = "FP"
            else:
                FN += 1
                outcome = "FN"

            row = {
                "Week": t,
                "Best_Threshold": float(best_thr),
                "Best_Score": float(best_score),
                "True_Label": true,
                "Pred_Label": pred,
                "Outcome": outcome,
                "thu_tue": float(test["thu/tue"].iloc[0]),
            }
            # Surface whichever hyperparameters the modeller was searching.
            for k, v in best_hp.items():
                row[f"Best_{k}"] = v
            rows.append(row)

        df_final = pd.DataFrame(rows)
        if df_final.empty:
            raise ValueError("Rolling classifier produced no rows. Check valid_weeks and data length.")

        df_final["week_period"] = weekly_full.index[df_final["Week"]]
        df_final["week_start_date"] = df_final["week_period"].dt.start_time
        df_final["correct"] = (df_final["True_Label"] == df_final["Pred_Label"]).astype(int)

        # Refit the modeller on the full dataset using the most recent best
        # hyperparameters. Convenient for downstream users who want to call
        # .predict() outside the pipeline. Falls back silently if best_hp
        # was never set.
        if best_hp is not None:
            self.modeller.fit(weekly_full[list(feature_columns)], weekly_full[target_column], **best_hp)

        self.confusion = {"TP": TP, "TN": TN, "FP": FP, "FN": FN}
        return df_final
