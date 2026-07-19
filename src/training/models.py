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


def to_xy(
    df: pl.DataFrame,
    features: list[str],
    categories: dict[str, list],
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Convert Polars to pandas while preserving trained categorical levels."""
    columns = [*features, "trips"] if "trips" in df.columns else list(features)
    pdf = df.select(columns).to_pandas()

    for column, cats in categories.items():
        pdf[column] = pd.Categorical(pdf[column], categories=cats)

    target = pdf["trips"] if "trips" in pdf.columns else None
    return pdf[features], target


def fit_lgbm(
    train_df: pl.DataFrame,
    features: list[str],
    params: dict,
    *,
    refit_full: bool = False,
) -> tuple[lgb.LGBMRegressor, dict[str, list], float]:
    """Fit LightGBM with time-based early stopping.

    ``refit_full=False`` preserves the Week 3 backtest behavior: fit on all
    months except the final validation month.

    ``refit_full=True`` first finds the best iteration using that validation
    month, then fits a fresh model with that number of trees on every usable
    row through the production cutoff. This prevents the production artifact
    from silently discarding its newest month of data.
    """
    lag_features = [
        feature for feature in features if feature.startswith(("lag_", "roll_"))
    ]
    usable = train_df.drop_nulls(subset=lag_features) if lag_features else train_df
    fit_df, valid_df = time_based_split(usable)
    if fit_df.is_empty() or valid_df.is_empty():
        raise ValueError("Training needs at least two calendar months.")

    categorical_columns = [
        column for column in CATEGORICAL_CANDIDATES if column in features
    ]
    validation_categories = {
        column: sorted(fit_df[column].unique().to_list())
        for column in categorical_columns
    }

    x_fit, y_fit = to_xy(fit_df, features, validation_categories)
    x_valid, y_valid = to_xy(valid_df, features, validation_categories)

    selection_model = lgb.LGBMRegressor(
        objective="poisson",
        verbose=-1,
        **params,
    )
    selection_model.fit(
        x_fit,
        y_fit,
        eval_set=[(x_valid, y_valid)],
        eval_metric="mae",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    val_predictions = selection_model.predict(x_valid)
    val_mae = float(np.abs(val_predictions - y_valid).mean())

    if not refit_full:
        return selection_model, validation_categories, val_mae

    final_categories = {
        column: sorted(usable[column].unique().to_list())
        for column in categorical_columns
    }
    x_all, y_all = to_xy(usable, features, final_categories)
    final_params = dict(params)
    best_iteration = selection_model.best_iteration_ or params["n_estimators"]
    final_params["n_estimators"] = int(best_iteration)

    final_model = lgb.LGBMRegressor(
        objective="poisson",
        verbose=-1,
        **final_params,
    )
    final_model.fit(x_all, y_all)
    return final_model, final_categories, val_mae


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

    def predict(self, train_df: pl.DataFrame, eval_df: pl.DataFrame) -> pl.Series:
        """Fit on train_df and predict eval_df."""
        model, categories, _ = fit_lgbm(
            train_df,
            self.features,
            self.params,
            refit_full=False,
        )
        self.last_model_ = model
        x_eval, _ = to_xy(eval_df, self.features, categories)
        predictions = model.predict(x_eval)
        return pl.Series(np.maximum(predictions, 0.0))
