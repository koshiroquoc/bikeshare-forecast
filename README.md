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

## Week 3 Day 1 — Default LightGBM result

The default LightGBM Poisson model outperforms both Week 2 baselines on the same 4 rolling-origin evaluation windows. It achieves a mean MAE of 1.101 and a MASE of 0.755, compared with 1.198 MAE and 0.820 MASE for the historical mean baseline, and 1.460 MAE and 1.000 MASE for the seasonal naive baseline.

This corresponds to a 24.6% MAE reduction relative to seasonal naive and an 8.1% MAE reduction relative to historical mean. The result is plausible: the model improves over both baselines, but not by an unrealistically large margin, so there is no immediate evidence of leakage.

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

## Week 3 feature iteration

All feature-iteration experiments use the same 4 rolling-origin evaluation windows and are tracked in MLflow.

| Variant | MAE mean ± std | RMSE mean | MASE | Decision | MLflow run |
|---|---:|---:|---:|---|---|
| LGBM default | 1.101 ± 0.283 | 1.832 | 0.755 | Baseline model | `<0fbe26b0b1d04fddb75910dbf67652dd>` |
| LGBM + station cluster | 1.101 ± 0.280 | 1.832 | 0.755 | Kill | `c757321b2c9a4be3a92bcff63f61f369` |
| LGBM + event flag | 1.103 ± 0.284 | 1.837 | 0.756 | Infrastructure only | `a69d728c7a2f416195bd998af338f395` |
| LGBM no weather | 1.154 ± 0.278 | 1.963 | 0.794 | Diagnostic only | `31a8c4494258492cbadee574e238be9f` |

**Cluster decision:** The station demand-shape cluster feature does not improve the default LightGBM model. Its MAE is effectively unchanged at 1.101, with the same RMSE and MASE as the default model. This suggests that LightGBM's native categorical `station_id` feature already captures most station-level structure. The cluster feature is therefore not included in the default model.

**Event decision:** The event feature is technically valid and has nonzero coverage in the evaluation windows, but it does not improve the overall backtest result. Event coverage is sparse: 3.2% of station-hours in 2026-03, 0.0% in 2026-04, 6.5% in 2026-05, and 3.3% in 2026-06. The event model has slightly worse MAE than the default model, 1.103 versus 1.101. The event infrastructure is kept for future analysis, but `is_major_event` is not included in the default model.

**Weather decision:** Weather features are important and should remain in the default model. Removing weather features increases MAE from 1.101 to 1.154 and RMSE from 1.832 to 1.963. This is a 0.053 MAE increase, or about a 4.8% degradation relative to the default LightGBM model. This supports the error-analysis finding that weather features are especially useful during rainy hours and still add value on the full rolling-origin backtest.

## Week 3 tuning and final model

A small LightGBM tuning sweep was run through the same 4 rolling-origin evaluation windows and tracked in MLflow. The search intentionally varied only a few high-impact hyperparameters: learning rate, tree complexity, and leaf regularization. The sweep used `n_estimators=500` for faster iteration; the final model is rerun with the default `n_estimators=2000` cap and early stopping.

| Candidate | MAE mean ± std | RMSE mean | MASE | Decision |
|---|---:|---:|---:|---|
| leaves_127 | 1.102 ± 0.281 | 1.831 | 0.756 | Best sweep candidate, not kept |
| lr_0_10 | 1.104 ± 0.283 | 1.845 | 0.757 | Not kept |
| min_child_500 | 1.108 ± 0.283 | 1.845 | 0.760 | Not kept |
| min_child_100 | 1.108 ± 0.282 | 1.844 | 0.760 | Not kept |
| min_child_20 | 1.108 ± 0.283 | 1.844 | 0.760 | Not kept |
| leaves_63 | 1.108 ± 0.283 | 1.844 | 0.760 | Default-equivalent sweep setting |
| lr_0_05 | 1.108 ± 0.283 | 1.844 | 0.760 | Default-equivalent sweep setting |
| lr_0_03 | 1.118 ± 0.285 | 1.866 | 0.767 | Not kept |
| leaves_31 | 1.119 ± 0.286 | 1.870 | 0.767 | Not kept |

**Selection rule:** The final configuration is the simplest configuration among the leading candidates that are not meaningfully separated by the standard deviation of MAE across backtest windows. This avoids overfitting the configuration to only four evaluation windows.

**Final decision:** The default LightGBM configuration is retained. Although `num_leaves=127` is the best candidate in the fast sweep, its MAE is 1.102, which is not meaningfully better than the default model's 1.101 MAE from the full Day 1 run. Increasing tree complexity is therefore not justified. The final model keeps `learning_rate=0.05`, `num_leaves=63`, and `n_estimators=2000`.

### Final model result

| Predictor | MAE mean ± std | RMSE mean | MASE | MLflow run |
|---|---:|---:|---:|---|
| LGBM final | 1.101 ± 0.283 | 1.832 | 0.755 | `2d509406102a484db1cc05c21e243903` |
| Historical mean | 1.198 ± 0.316 | 2.267 | 0.820 | `2d509406102a484db1cc05c21e243903` |
| Seasonal naive | 1.460 ± 0.385 | 2.673 | 1.000 | `2d509406102a484db1cc05c21e243903` |

### Week 3 model design decisions

| Decision | Rationale |
|---|---|
| LightGBM uses a Poisson objective | The target is a non-negative station-hour trip count and the dataset is zero-inflated. A Poisson objective matches count forecasting better than squared-error regression and naturally produces non-negative predictions. |
| Early stopping uses a time-based split | The last month of each training window is used for validation. Random validation would leak nearby future observations into training and produce overly optimistic results. |
| `station_id` is handled as a native categorical feature | This lets LightGBM learn station-level effects without one-hot encoding hundreds of stations. Unknown stations at prediction time become missing categorical values rather than crashing the model. |
| All variants use the same rolling-origin backtest | Baselines, default LightGBM, feature variants, and tuning candidates are compared on the same evaluation windows. This prevents misleading comparisons across different splits. |
| Feature decisions are made using ΔMAE and 1-std noise awareness | Improvements smaller than the window-to-window MAE variation are treated as noise. This is why the cluster feature and the best fast tuning candidate are not kept. |
| Weather features are retained | The no-weather variant increases MAE from 1.101 to 1.154 and RMSE from 1.832 to 1.963, showing that weather adds value beyond lag features. |
| Cluster feature is rejected | The station-cluster variant has essentially the same MAE as the default model, suggesting that native `station_id` already captures most station-level structure. |
| Event feature is kept as infrastructure only | Event coverage is sparse in the current backtest windows and the event variant slightly worsens MAE. The code remains available for future analysis, but the feature is not part of the default model. |
| The final config favors simplicity over tuning noise | The best fast tuning candidate is not meaningfully better than the default full run, so the default configuration is retained rather than increasing complexity. |
 
### Final model feature importance

The final LightGBM model is driven primarily by demand-history features rather than weather alone. The top gain-based features are `roll_mean_28d`, `roll_mean_7d`, `station_id`, `lag_168`, and `day_of_week`.

This pattern is consistent with the problem structure. Bikeshare demand is highly persistent at the station-hour level, so recent rolling averages and weekly lag features provide the strongest signal. `station_id` also has high importance, which supports the Day 4 result that an additional station-cluster feature is redundant: LightGBM already captures most station-level structure directly from the categorical station identifier.

Calendar features such as `day_of_week`, `is_weekend`, and `month_cos` also contribute meaningfully, reflecting weekly and seasonal demand cycles. Weather features such as `wind_speed_10m` and `snowfall` appear lower in the gain ranking, but they still matter operationally: the Day 4 no-weather experiment showed that removing weather features increases MAE from 1.101 to 1.154, confirming that weather improves the model even if demand-history features dominate overall gain.

Overall, the feature-importance profile supports the interpretation that the model improves over seasonal naive not by ignoring history, but by learning when a weekly copy should be adjusted using station identity, smoother demand trends, calendar context, and weather.

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