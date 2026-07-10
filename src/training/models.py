"""LightGBM predictors for station-hour bikeshare demand forecasting.

Design decisions:
1. Poisson objective: the target is a non-negative count and station-hour demand is sparse.
2. Time-based early stopping: the last month of train_df is validation.
3. Native categorical station_id: LightGBM handles station effects without one-hot encoding.
"""

from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl

CATEGORICAL_CANDIDATES = ["station_id", "cluster"]


def time_based_split(train_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Use the last calendar month of train_df as validation data."""
    last_hour = train_df["hour"].max()
    valid_start = datetime(last_hour.year, last_hour.month, 1)

    fit_df = train_df.filter(pl.col("hour") < valid_start)
    valid_df = train_df.filter(pl.col("hour") >= valid_start)

    return fit_df, valid_df


class LGBMPredictor:
    """LightGBM predictor with the same interface as Week 2 baselines."""

    def __init__(
        self,
        features: list[str],
        params: dict,
        name: str = "lgbm",
    ) -> None:
        self.features = features
        self.params = dict(params)
        self.name = name
        self.last_model_ = None

    def _to_pandas_xy(
        self,
        df: pl.DataFrame,
        categories: dict[str, list],
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Convert Polars data to pandas and align categorical levels."""
        pdf = df.select([*self.features, "trips"]).to_pandas()

        for column, cats in categories.items():
            pdf[column] = pd.Categorical(pdf[column], categories=cats)

        return pdf[self.features], pdf["trips"]

    def predict(self, train_df: pl.DataFrame, eval_df: pl.DataFrame) -> pl.Series:
        """Fit on train_df and predict eval_df."""
        lag_features = [
            feature
            for feature in self.features
            if feature.startswith(("lag_", "roll_"))
        ]

        usable_train = (
            train_df.drop_nulls(subset=lag_features) if lag_features else train_df
        )

        fit_df, valid_df = time_based_split(usable_train)

        categorical_columns = [
            column for column in CATEGORICAL_CANDIDATES if column in self.features
        ]

        categories = {
            column: sorted(fit_df[column].unique().to_list())
            for column in categorical_columns
        }

        x_fit, y_fit = self._to_pandas_xy(fit_df, categories)
        x_valid, y_valid = self._to_pandas_xy(valid_df, categories)
        x_eval, _ = self._to_pandas_xy(eval_df, categories)

        model = lgb.LGBMRegressor(
            objective="poisson",
            verbose=-1,
            **self.params,
        )

        model.fit(
            x_fit,
            y_fit,
            eval_set=[(x_valid, y_valid)],
            eval_metric="mae",
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        self.last_model_ = model

        predictions = model.predict(x_eval)
        return pl.Series(np.maximum(predictions, 0.0))
