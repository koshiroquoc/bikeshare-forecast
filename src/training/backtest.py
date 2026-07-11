"""Rolling-origin backtest framework for station-hour demand forecasting.

A predictor is any object with:
- name: str
- predict(train_df, eval_df) -> pl.Series

This keeps the backtest model-agnostic: baselines, LightGBM, and future models
all use the same evaluation path.
"""

from datetime import timedelta

import polars as pl


def mae(y: pl.Series, yhat: pl.Series) -> float:
    """Mean absolute error."""
    return (y - yhat).abs().mean()


def rmse(y: pl.Series, yhat: pl.Series) -> float:
    """Root mean squared error."""
    return ((y - yhat) ** 2).mean() ** 0.5


class SeasonalNaive:
    """Predict demand using the same station-hour from one week earlier."""

    name = "seasonal_naive"

    def predict(self, train_df: pl.DataFrame, eval_df: pl.DataFrame) -> pl.Series:
        """Use leak-safe lag_168 as the forecast."""
        return eval_df["lag_168"].fill_null(0)


class HistoricalMean:
    """Predict using recent mean demand by station, day of week, and hour."""

    name = "historical_mean"

    def predict(self, train_df: pl.DataFrame, eval_df: pl.DataFrame) -> pl.Series:
        """Use the previous 8 weeks before the evaluation cutoff."""
        cutoff = train_df["hour"].max()
        recent = train_df.filter(pl.col("hour") > cutoff - timedelta(weeks=8))

        keys = ["station_id", "day_of_week", "hour_of_day"]

        profile = recent.group_by(keys).agg(
            pl.col("trips").mean().alias("profile_mean")
        )

        station_mean = recent.group_by("station_id").agg(
            pl.col("trips").mean().alias("station_mean")
        )

        return (
            eval_df.join(profile, on=keys, how="left")
            .join(station_mean, on="station_id", how="left")
            .select(
                pl.coalesce(
                    pl.col("profile_mean"),
                    pl.col("station_mean"),
                    pl.lit(0.0),
                ).alias("prediction")
            )
            .to_series()
        )


def eval_months(df: pl.DataFrame, n_windows: int) -> list[str]:
    """Return the last n full calendar months available in the feature table."""
    months = (
        df.select(pl.col("hour").dt.strftime("%Y-%m").alias("month"))
        .unique()
        .sort("month")["month"]
        .to_list()
    )

    last_hour = df["hour"].max()

    month_start = last_hour.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end_of_month = (month_start + timedelta(days=32)).replace(day=1) - timedelta(
        hours=1
    )

    if last_hour < end_of_month:
        months = months[:-1]

    return months[-n_windows:]


def run_backtest(
    features_df: pl.DataFrame,
    predictors: list,
    n_windows: int,
) -> pl.DataFrame:
    """Run rolling-origin backtest.

    For each evaluation month:
    - train = all rows before that month
    - eval = rows inside that month
    """
    rows = []
    df = features_df.with_columns(
        pl.col("hour").dt.strftime("%Y-%m").alias("_eval_month")
    )

    for month in eval_months(features_df, n_windows):
        train_df = df.filter(pl.col("_eval_month") < month)
        eval_df = df.filter(pl.col("_eval_month") == month)

        for predictor in predictors:
            yhat = predictor.predict(train_df, eval_df)

            rows.append(
                {
                    "predictor": predictor.name,
                    "window": month,
                    "mae": mae(eval_df["trips"], yhat),
                    "rmse": rmse(eval_df["trips"], yhat),
                    "n": eval_df.height,
                }
            )

    return pl.DataFrame(rows)


def summarize(results: pl.DataFrame) -> pl.DataFrame:
    """Summarize backtest results and compute MASE against seasonal naive."""
    naive = results.filter(pl.col("predictor") == "seasonal_naive").select(
        "window",
        pl.col("mae").alias("naive_mae"),
    )

    with_mase = results.join(naive, on="window", how="left").with_columns(
        (pl.col("mae") / pl.col("naive_mae")).alias("mase")
    )

    return (
        with_mase.group_by("predictor")
        .agg(
            pl.col("mae").mean().round(3).alias("mae_mean"),
            pl.col("mae").std().round(3).alias("mae_std"),
            pl.col("rmse").mean().round(3).alias("rmse_mean"),
            pl.col("mase").mean().round(3).alias("mase_mean"),
        )
        .sort("mae_mean")
    )


def prediction_table(
    features_df: pl.DataFrame,
    predictors: list,
    window_month: str,
) -> pl.DataFrame:
    """Build detailed prediction table for one evaluation month.

    Returns one row per station-hour with actual trips and one prediction column
    per predictor. This table is the source for error analysis slices.
    """
    df = features_df.with_columns(
        pl.col("hour").dt.strftime("%Y-%m").alias("_eval_month")
    )

    train_df = df.filter(pl.col("_eval_month") < window_month)
    eval_df = df.filter(pl.col("_eval_month") == window_month)

    out = eval_df.select("station_id", "hour", "trips")

    for predictor in predictors:
        predictions = predictor.predict(train_df, eval_df)
        out = out.with_columns(predictions.alias(f"pred_{predictor.name}"))

    return out
