# Divvy Demand Forecasting

Dự báo số lượt mượn xe theo trạm-giờ cho 24h tới, phục vụ bài toán rebalancing của hệ thống bike-share Divvy (Chicago).

## Design decisions

week 1:
1. Cleaning rules 60s / 24h / missing station — with the 19,74% removed per rule (section 3–4).
2. Station mapping: 1,659 IDs remapped, ambiguous pairs deliberately left unmapped (section 5).
3. Scope: `top_n_stations = 200` covering 65% of last-12-month volume (section 8) — update `config.yaml` now.
4. Zero-inflation: 52% of station-hours are zero in scope — the evidence for the Poisson objective in week 3.
5. Timezone sanity + weather join ≥95% both asserted in-code (sections 6–7).

week 2:
- **Full station-hour grid with zero filling.** The modeling table includes every selected station for every hour in the observed time range, and missing station-hours are filled with `trips = 0`. This prevents the model from learning only from hours with observed trips and is protected by the full-grid unit test.

- **Station lifetime trimming.** Station-hour rows before a station's first observed month are removed. This avoids teaching the model that a station had zero demand before it existed.

- **All target-derived features use at least a 24-hour lag.** Because the forecast horizon is 24 hours, lag features must not use information from the previous 1–23 hours. The leakage guard test corrupts the interval after `H-24` and verifies that features at `H` do not change.

- **Same-hour rolling means instead of ordinary rolling windows.** `roll_mean_7d` uses the same hour from previous days, such as `t-24h`, `t-48h`, and so on. A normal rolling window over the previous 168 hours would leak recent target values that are not available for a 24-hour-ahead forecast.

- **One feature code path for training and serving.** Calendar, weather, station, and lag features are implemented in reusable processing functions rather than notebook-only logic. This reduces train/serve skew because future batch predictions can call the same feature code used for training.

- **Weather actuals for training and forecast weather for serving.** Historical training uses observed weather, while future serving will use forecast weather with the same schema. This is an intentional source of train/serve difference and should be monitored later.

- **DST is ignored in the first version.** The station-hour grid uses naive local hourly timestamps, which can create a small number of imperfect hours around daylight saving time transitions. This affects only a tiny fraction of annual hours and is acceptable for the first modeling iteration.

- **Backtest is model-agnostic.** A predictor only needs a `name` and a `predict(train_df, eval_df)` method. This allows seasonal naive, historical mean, and future LightGBM models to use the same rolling-origin evaluation framework.

## Baseline results

Rolling-origin backtest uses the last 4 full calendar months as evaluation windows: March 2026 through June 2026. For each window, the training set contains all station-hour rows before the evaluation month, and the evaluation set contains only rows inside that month.

| Baseline | MAE mean ± std | RMSE mean | MASE |
|---|---:|---:|---:|
| Historical mean | 1.198 ± 0.316 | 2.267 | 0.820 |
| Seasonal naive | 1.460 ± 0.385 | 2.673 | 1.000 |

**Conclusion:** The historical mean baseline performs better than the seasonal naive baseline, with lower MAE, lower RMSE, and a MASE of 0.820. This means the historical mean reduces error by about 18% relative to the weekly seasonal naive benchmark. The MAE values are within the expected range for top-200 station-hour demand forecasting, so the baseline results look reasonable and do not suggest obvious leakage or grid construction errors.

## Week 2 design decisions

- **Full station-hour grid with zero filling.** The modeling table includes every selected station for every hour in the observed time range, and missing station-hours are filled with `trips = 0`. This prevents the model from learning only from hours with observed trips and is protected by the full-grid unit test.

- **Station lifetime trimming.** Station-hour rows before a station's first observed month are removed. This avoids teaching the model that a station had zero demand before it existed.

- **All target-derived features use at least a 24-hour lag.** Because the forecast horizon is 24 hours, lag features must not use information from the previous 1 to 23 hours. The leakage guard test corrupts the interval after `H-24` and verifies that features at `H` do not change.

- **Same-hour rolling means instead of ordinary rolling windows.** `roll_mean_7d` uses the same hour from previous days, such as `t-24h`, `t-48h`, and so on. A normal rolling window over the previous 168 hours would leak recent target values that are not available for a 24-hour-ahead forecast.

- **One feature code path for training and serving.** Calendar, weather, station, and lag features are implemented in reusable processing functions rather than notebook-only logic. This reduces train/serve skew because future batch predictions can call the same feature code used for training.

- **Weather actuals for training and forecast weather for serving.** Historical training uses observed weather, while future serving will use forecast weather with the same schema. This is an intentional source of train/serve difference and should be monitored later.

- **DST is ignored in the first version.** The station-hour grid uses naive local hourly timestamps, which can create a small number of imperfect hours around daylight saving time transitions. This affects only a tiny fraction of annual hours and is acceptable for the first modeling iteration.

- **Backtest is model-agnostic.** A predictor only needs a `name` and a `predict(train_df, eval_df)` method. This allows seasonal naive, historical mean, and future LightGBM models to use the same rolling-origin evaluation framework.

## Data products

The processed data products are generated by running:

```bash
make features
```

These files are written to `data/processed/` and are not committed to Git because they are reproducible from raw data and project code.

### `features.parquet`

Station-hour modeling table for the selected top-200 station scope.

Core columns:

| Column | Description |
|---|---|
| `station_id` | Canonical station identifier after station mapping |
| `hour` | Hourly timestamp |
| `trips` | Number of trips starting from the station during that hour |
| `hour_of_day` | Hour of day |
| `day_of_week` | Day of week |
| `month` | Calendar month |
| `is_weekend` | Weekend indicator |
| `is_holiday` | Illinois/US holiday indicator |
| `hour_sin`, `hour_cos` | Cyclical hour encoding |
| `month_sin`, `month_cos` | Cyclical month encoding |
| `temperature_2m` | Hourly temperature |
| `precipitation` | Hourly precipitation |
| `snowfall` | Hourly snowfall |
| `wind_speed_10m` | Hourly wind speed |
| `relative_humidity_2m` | Hourly relative humidity |
| `lat`, `lng` | Station coordinates |
| `lag_24`, `lag_48`, `lag_168` | Leak-safe historical demand lags |
| `roll_mean_7d`, `roll_mean_28d` | Leak-safe same-hour rolling demand means |

### `station_master.parquet`

Station-level reference table for all cleaned stations, not only the top-200 modeling scope.

| Column | Description |
|---|---|
| `station_id` | Canonical station identifier |
| `name` | Most common station name |
| `lat` | Median latitude |
| `lng` | Median longitude |
| `total_trips` | Total cleaned trips starting from the station |
| `first_month` | First observed month |
| `last_month` | Last observed month |

### `station_month_panel.parquet`

Station-month panel for downstream station analysis.

| Column | Description |
|---|---|
| `station_id` | Canonical station identifier |
| `month` | Calendar month |
| `member_trips` | Member trips in that station-month |
| `casual_trips` | Casual trips in that station-month |
| `total_trips` | Total trips in that station-month |

These schemas are treated as a contract for downstream analysis and should not be renamed casually.


**Conclusion:** The historical mean baseline performs better than the seasonal naive baseline, with lower MAE, lower RMSE, and a MASE of 0.820. This means the historical mean reduces error by about 18% relative to the weekly seasonal naive benchmark. The MAE values are within the expected range for top-200 station-hour demand forecasting, so the baseline results look reasonable and do not suggest obvious leakage or grid construction errors.