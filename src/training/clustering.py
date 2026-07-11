"""Station clustering features for bikeshare demand forecasting.

Stations are clustered by demand-shape profile, not raw demand scale.
This tests whether station profile groups add information beyond the native
categorical station_id feature used by LightGBM.
"""

import polars as pl
from sklearn.cluster import KMeans


def station_profiles(grid: pl.DataFrame) -> pl.DataFrame:
    """Build one normalized demand-shape vector per station.

    Each station is represented by a 48-dimensional profile:
    - 24 hours for weekdays
    - 24 hours for weekends
    """
    profile = (
        grid.with_columns(
            pl.col("hour").dt.hour().alias("hour_of_day"),
            (pl.col("hour").dt.weekday() >= 6).cast(pl.Int8).alias("is_weekend"),
        )
        .group_by(["station_id", "is_weekend", "hour_of_day"])
        .agg(pl.col("trips").mean().alias("mean_trips"))
        .with_columns(
            (
                pl.col("is_weekend").cast(pl.Utf8)
                + "_"
                + pl.col("hour_of_day").cast(pl.Utf8)
            ).alias("slot")
        )
        .pivot(
            on="slot",
            index="station_id",
            values="mean_trips",
        )
        .fill_null(0.0)
    )

    value_columns = [column for column in profile.columns if column != "station_id"]
    total_profile_demand = pl.sum_horizontal(
        [pl.col(column) for column in value_columns]
    )

    return profile.with_columns(
        [
            pl.when(total_profile_demand > 0)
            .then(pl.col(column) / total_profile_demand)
            .otherwise(0.0)
            .alias(column)
            for column in value_columns
        ]
    )


def cluster_stations(
    grid: pl.DataFrame,
    k: int,
    seed: int = 42,
) -> pl.DataFrame:
    """Cluster stations by normalized demand-shape profile."""
    profiles = station_profiles(grid)
    value_columns = [column for column in profiles.columns if column != "station_id"]

    labels = KMeans(
        n_clusters=k,
        random_state=seed,
        n_init=10,
    ).fit_predict(profiles.select(value_columns).to_numpy())

    return profiles.select("station_id").with_columns(
        pl.Series("cluster", labels.astype(str))
    )


def add_cluster_feature(
    df: pl.DataFrame,
    clusters: pl.DataFrame,
) -> pl.DataFrame:
    """Join cluster labels onto a feature table."""
    return df.join(clusters, on="station_id", how="left").with_columns(
        pl.col("cluster").fill_null("unknown")
    )
