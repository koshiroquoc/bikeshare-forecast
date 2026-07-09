"""Week 2 processing tests: cleaning, grid construction, and leakage guards."""

from datetime import datetime, timedelta

import polars as pl

from src.processing import aggregate, cleaning, features

CFG = {
    "scope": {
        "min_trip_seconds": 60,
        "max_trip_hours": 24,
        "top_n_stations": 2,
    }
}

T0 = datetime(2025, 5, 1, 8, 0)


def _trip(
    start: datetime,
    duration_seconds: int,
    station: str | None = "A",
    name: str = "Station A",
    lat: float = 41.9,
    lng: float = -87.6,
    member: str = "member",
) -> dict:
    return {
        "started_at": start,
        "ended_at": start + timedelta(seconds=duration_seconds),
        "start_station_id": station,
        "start_station_name": name,
        "start_lat": lat,
        "start_lng": lng,
        "member_casual": member,
    }


def make_raw() -> pl.LazyFrame:
    rows = [
        _trip(T0, 600),
        _trip(T0, 30),
        _trip(T0, -100),
        _trip(T0, 26 * 3600),
        _trip(T0, 600, station=None),
        _trip(T0, 600, station="OLD_ID"),
    ]

    return pl.DataFrame(rows).lazy()


MAPPING = pl.DataFrame(
    {
        "raw_station_id": ["OLD_ID"],
        "canonical_station_id": ["A"],
    }
)


def test_removal_report_counts_each_rule() -> None:
    report = cleaning.removal_report(make_raw(), CFG)

    assert report["total"] == 6
    assert report["missing_station"] == 1
    assert report["too_short_or_negative"] == 2
    assert report["too_long"] == 1


def test_clean_trips_filters_and_maps_station_ids() -> None:
    out = cleaning.clean_trips(make_raw(), CFG, MAPPING).collect()

    assert out.height == 2
    assert out["station_id"].to_list() == ["A", "A"]


def make_clean_two_stations() -> pl.LazyFrame:
    rows = []

    for day in range(3):
        for hour in [8, 17]:
            t = datetime(2025, 5, 1 + day, hour, 15)
            rows.append(_trip(t, 600, station="A"))
            rows.append(_trip(t, 900, station="A"))

    rows.append(
        _trip(
            datetime(2025, 5, 2, 12, 5),
            600,
            station="B",
            name="Station B",
        )
    )

    return cleaning.clean_trips(pl.DataFrame(rows).lazy(), CFG, MAPPING)


def test_full_grid_shape_zeros_and_trip_conservation() -> None:
    clean = make_clean_two_stations()
    counts = aggregate.aggregate_station_hour(clean, ["A", "B"])
    grid = aggregate.build_full_grid(counts, ["A", "B"])

    n_hours = grid["hour"].n_unique()

    assert grid.height == 2 * n_hours
    assert grid.filter(pl.col("trips") == 0).height > 0
    assert grid["trips"].sum() == 13

    b_at_noon = grid.filter(
        (pl.col("station_id") == "B") & (pl.col("hour") == datetime(2025, 5, 2, 12))
    )

    assert b_at_noon["trips"][0] == 1


def test_trim_to_station_lifetime() -> None:
    rows = [
        _trip(datetime(2025, 5, 1, 8), 600, station="A"),
        _trip(datetime(2025, 6, 15, 8), 600, station="A"),
        _trip(datetime(2025, 6, 10, 9), 600, station="LATE", name="Late Station"),
    ]

    clean = cleaning.clean_trips(pl.DataFrame(rows).lazy(), CFG, MAPPING)
    counts = aggregate.aggregate_station_hour(clean, ["A", "LATE"])
    grid = aggregate.build_full_grid(counts, ["A", "LATE"])
    trimmed = aggregate.trim_to_station_lifetime(grid, clean)

    late_min_hour = trimmed.filter(pl.col("station_id") == "LATE")["hour"].min()
    a_min_hour = trimmed.filter(pl.col("station_id") == "A")["hour"].min()

    assert late_min_hour >= datetime(2025, 6, 1)
    assert a_min_hour < datetime(2025, 6, 1)


def make_grid_30_days() -> pl.DataFrame:
    hours = pl.datetime_range(
        datetime(2025, 4, 1),
        datetime(2025, 4, 30, 23),
        interval="1h",
        eager=True,
    )

    return pl.DataFrame(
        {
            "station_id": ["A"] * len(hours),
            "hour": hours,
            "trips": list(range(len(hours))),
        }
    ).sort(["station_id", "hour"])


LAG_HOURS = [24, 48, 168]
ROLLING_DAYS = [7]
TARGET_DERIVED_COLUMNS = ["lag_24", "lag_48", "lag_168", "roll_mean_7d"]


def test_leakage_guard_features_blind_after_h_minus_24() -> None:
    grid = make_grid_30_days()
    h = datetime(2025, 4, 25, 17)

    before = features.add_lag_features(
        grid,
        LAG_HOURS,
        ROLLING_DAYS,
    ).filter(pl.col("hour") == h)

    corrupted = grid.with_columns(
        pl.when(pl.col("hour") > h - timedelta(hours=24))
        .then(999_999)
        .otherwise(pl.col("trips"))
        .alias("trips")
    )

    after = features.add_lag_features(
        corrupted,
        LAG_HOURS,
        ROLLING_DAYS,
    ).filter(pl.col("hour") == h)

    for column in TARGET_DERIVED_COLUMNS:
        assert before[column][0] == after[column][0], (
            f"{column} leaked information after H-24."
        )


def test_lag_24_exact_value() -> None:
    out = features.add_lag_features(make_grid_30_days(), LAG_HOURS, ROLLING_DAYS)
    h = datetime(2025, 4, 10, 9)

    expected = out.filter(pl.col("hour") == h - timedelta(hours=24))["trips"][0]
    actual = out.filter(pl.col("hour") == h)["lag_24"][0]

    assert actual == expected


def test_lag_does_not_cross_stations() -> None:
    hours = pl.datetime_range(
        datetime(2025, 4, 1),
        datetime(2025, 4, 5, 23),
        interval="1h",
        eager=True,
    )

    grid = pl.concat(
        [
            pl.DataFrame(
                {
                    "station_id": ["A"] * len(hours),
                    "hour": hours,
                    "trips": [1000] * len(hours),
                }
            ),
            pl.DataFrame(
                {
                    "station_id": ["B"] * len(hours),
                    "hour": hours,
                    "trips": [1] * len(hours),
                }
            ),
        ]
    ).sort(["station_id", "hour"])

    out = features.add_lag_features(grid, [24], [])
    b_lags = out.filter((pl.col("station_id") == "B") & pl.col("lag_24").is_not_null())[
        "lag_24"
    ]

    assert set(b_lags.to_list()) == {1}
