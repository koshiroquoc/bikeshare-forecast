import io
import zipfile

import polars as pl
import pytest

from src.ingestion import divvy

CSV_HEADER = (
    "ride_id,rideable_type,started_at,ended_at,start_station_name,start_station_id,"
    "end_station_name,end_station_id,start_lat,start_lng,end_lat,end_lng,member_casual\n"
)

ROW_PLAIN = (
    "A1,classic_bike,2024-05-01 08:15:00,2024-05-01 08:30:00,"
    "Clark St,13045,State St,TA1309000030,41.9,-87.6,41.91,-87.62,member\n"
)

ROW_MILLIS = (
    "A2,electric_bike,2024-05-01 09:00:00.123,2024-05-01 09:20:00.456,"
    "Lake St,chargingstx07,,,41.88,-87.63,,,casual\n"
)


def make_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("202405-divvy-tripdata.csv", CSV_HEADER + ROW_PLAIN + ROW_MILLIS)
        zf.writestr("__MACOSX/junk.csv", "trash")
    return buf.getvalue()


def test_month_from_key():
    assert divvy.month_from_key("202405-divvy-tripdata.zip") == "2024-05"
    assert divvy.month_from_key("Divvy_Trips_2020_Q1.zip") is None
    assert divvy.month_from_key("index.html") is None


def test_read_trip_zip_handles_traps():
    df = divvy.read_trip_zip(make_zip(), "test.zip")

    assert df.height == 2
    assert df.schema["start_station_id"] == pl.Utf8
    assert df.schema["end_station_id"] == pl.Utf8
    assert df["started_at"].dt.hour().to_list() == [8, 9]


def test_validate_schema_fails_loud():
    df = divvy.read_trip_zip(make_zip(), "test.zip").rename({"start_station_id": "x"})

    with pytest.raises(ValueError, match="start_station_id"):
        divvy.validate_schema(df, "bad.csv")
