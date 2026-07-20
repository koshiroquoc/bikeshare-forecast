# Divvy Demand Forecasting

End-to-end ML system that forecasts station-hour bike demand 24 hours ahead for Chicago's Divvy bike-share network, supporting the rebalancing problem (deciding where to move bikes before demand hits).

## Headline results

Final LightGBM (Poisson) model vs. baselines, evaluated on 4 rolling-origin monthly windows (Mar–Jun 2026):

| Predictor | MAE mean ± std | RMSE mean | MASE | MLflow run |
|---|---:|---:|---:|---|
| **LGBM final** | **1.101 ± 0.283** | **1.832** | **0.755** | `2d509406102a484db1cc05c21e243903` |
| Historical mean | 1.198 ± 0.316 | 2.267 | 0.820 | `2d509406102a484db1cc05c21e243903` |
| Seasonal naive | 1.460 ± 0.385 | 2.673 | 1.000 | `2d509406102a484db1cc05c21e243903` |

- **24.6% MAE reduction** vs. seasonal naive, **8.1%** vs. historical mean.
- The margin is plausible rather than suspiciously large — no evidence of leakage, which is additionally protected by an explicit leakage guard test.

## Tech stack

| Layer | Tool |
|---|---|
| Data processing | Polars |
| Modeling | LightGBM (Poisson objective) |
| Experiment tracking | MLflow 3.x (SQLite backend) |
| Testing | pytest (full-grid, leakage guard, train/serve consistency) |
| Package management | uv |
| Serving | FastAPI over leak-safe, precomputed daily forecasts |
| Orchestration | Prefect 3 |
| Monitoring | Evidently 0.7 + machine-readable performance/drift gates |
| Delivery | Docker Compose + GitHub Actions |

## Problem setup

- **Target:** trips starting from each station in each hour, forecast horizon 24 hours.
- **Scope:** top 200 stations (`top_n_stations = 200` in `config.yaml`), covering 65% of last-12-month volume.
- **Evaluation:** rolling-origin backtest over the last 4 full calendar months. For each window, training uses all station-hour rows before the evaluation month; evaluation uses only rows inside that month. All models — baselines, feature variants, tuning candidates — are compared on the same windows.

## Design decisions

Grouped by theme rather than chronology. Each decision is enforced in code (tests or assertions) where possible.

### Leakage discipline

- **All target-derived features use at least a 24-hour lag.** Because the forecast horizon is 24 hours, lag features must not use information from the previous 1–23 hours. The leakage guard test corrupts the interval after `H-24` and verifies that features at `H` do not change.
- **Same-hour rolling means instead of ordinary rolling windows.** `roll_mean_7d` uses the same hour from previous days (`t-24h`, `t-48h`, …). A normal rolling window over the previous 168 hours would leak recent target values that are not available for a 24-hour-ahead forecast.
- **Early stopping uses a time-based split.** The last month of each training window is used for validation. Random validation would leak nearby future observations into training and produce overly optimistic results.
- **Weather actuals for training, forecast weather for serving.** Historical training uses observed weather; future serving will use forecast weather with the same schema. This is an intentional, monitored source of train/serve difference.

### Data construction

- **Cleaning rules: 60s / 24h / missing station**, removing 19.74% of raw trips (details in the Week 1 notebook, sections 3–4).
- **Station mapping:** 1,659 IDs remapped to canonical stations; ambiguous pairs deliberately left unmapped.
- **Full station-hour grid with zero filling.** The modeling table includes every selected station for every hour in the observed range; missing station-hours are filled with `trips = 0`. This prevents the model from learning only from hours with observed trips and is protected by the full-grid unit test.
- **Station lifetime trimming.** Rows before a station's first observed month are removed, so the model never learns that a station had zero demand before it existed.
- **Zero-inflation evidence:** 52% of in-scope station-hours are zero — the justification for the Poisson objective.
- **Timezone sanity and weather-join coverage (≥95%) asserted in code.**
- **DST is ignored in the first iteration.** The grid uses naive local hourly timestamps, which affects only a tiny fraction of annual hours around DST transitions and is acceptable for v1.

### Modeling and evaluation

- **Poisson objective.** The target is a non-negative, zero-inflated count; Poisson matches count forecasting better than squared-error regression and naturally produces non-negative predictions.
- **`station_id` as a native categorical feature.** LightGBM learns station-level effects without one-hot encoding hundreds of stations; unknown stations at prediction time become missing categorical values rather than crashing the model.
- **Model-agnostic backtest.** A predictor only needs a `name` and a `predict(train_df, eval_df)` method, so baselines and LightGBM variants share the same rolling-origin evaluation framework.
- **Feature decisions use ΔMAE with 1-std noise awareness.** Improvements smaller than window-to-window MAE variation are treated as noise — this is why the cluster feature and the best fast-sweep candidate were not kept.
- **One feature code path for training and serving.** Calendar, weather, station, and lag features live in reusable processing functions rather than notebook-only logic, reducing train/serve skew for future batch predictions.

## Experiments

### Baselines

| Baseline | MAE mean ± std | RMSE mean | MASE |
|---|---:|---:|---:|
| Historical mean | 1.198 ± 0.316 | 2.267 | 0.820 |
| Seasonal naive | 1.460 ± 0.385 | 2.673 | 1.000 |

The historical mean beats the weekly seasonal naive by about 18% MAE. Values are within the expected range for top-200 station-hour demand, with no signs of leakage or grid construction errors.

### Feature iteration

All experiments use the same 4 evaluation windows and are tracked in MLflow.

| Variant | MAE mean ± std | RMSE mean | MASE | Decision | MLflow run |
|---|---:|---:|---:|---|---|
| LGBM default | 1.101 ± 0.283 | 1.832 | 0.755 | Baseline model | `0fbe26b0b1d04fddb75910dbf67652dd` |
| LGBM + station cluster | 1.101 ± 0.280 | 1.832 | 0.755 | Kill | `c757321b2c9a4be3a92bcff63f61f369` |
| LGBM + event flag | 1.103 ± 0.284 | 1.837 | 0.756 | Infrastructure only | `a69d728c7a2f416195bd998af338f395` |
| LGBM no weather | 1.154 ± 0.278 | 1.963 | 0.794 | Diagnostic only | `31a8c4494258492cbadee574e238be9f` |

- **Cluster — rejected.** MAE effectively unchanged at 1.101; LightGBM's native categorical `station_id` already captures most station-level structure.
- **Event flag — infrastructure only.** Technically valid with nonzero coverage in the eval windows, but coverage is sparse (3.2% of station-hours in 2026-03, 0.0% in 2026-04, 6.5% in 2026-05, 3.3% in 2026-06) and MAE is slightly worse (1.103 vs. 1.101). The event pipeline (`chicago_events.csv`, `src/training/events.py`, tests) is kept for future analysis but `is_major_event` is not in the default model.
- **Weather — retained.** Removing weather increases MAE from 1.101 to 1.154 (+4.8%) and RMSE from 1.832 to 1.963, consistent with the error-analysis finding that weather features are especially useful during rainy hours.

### Hyperparameter tuning

A small sweep over high-impact hyperparameters (learning rate, tree complexity, leaf regularization) using `n_estimators=500` for fast iteration; the final model reruns with the default `n_estimators=2000` cap and early stopping.

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

**Selection rule:** the simplest configuration among candidates not meaningfully separated by the MAE standard deviation across backtest windows — this avoids overfitting the configuration to only four evaluation windows.

**Final decision:** the default configuration is retained (`learning_rate=0.05`, `num_leaves=63`, `n_estimators=2000` with early stopping). `num_leaves=127` reached 1.102 MAE, not meaningfully better than the default's 1.101, so extra tree complexity is not justified.

### Feature importance

The final model is driven primarily by demand-history features. Top gain-based features: `roll_mean_28d`, `roll_mean_7d`, `station_id`, `lag_168`, `day_of_week`.

This matches the problem structure: bikeshare demand is highly persistent at the station-hour level, so rolling averages and weekly lags carry the strongest signal. The high importance of `station_id` supports the cluster-feature rejection — LightGBM already captures station-level structure directly from the categorical identifier.

Calendar features (`day_of_week`, `is_weekend`, `month_cos`) contribute meaningfully. Weather features (`wind_speed_10m`, `snowfall`) rank lower in gain but matter operationally: the no-weather ablation showed a 1.101 → 1.154 MAE degradation.

Overall, the model improves over seasonal naive not by ignoring history, but by learning when a weekly copy should be adjusted using station identity, smoother demand trends, calendar context, and weather.

## Data products

Generated by:

```bash
make features
```

Files are written to `data/processed/` and are not committed to Git because they are reproducible from raw data and project code. These schemas are treated as a contract for downstream analysis and should not be renamed casually.

### `features.parquet`

Station-hour modeling table for the top-200 station scope.

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

Station-level reference table for all cleaned stations (not only the top-200 scope).

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

## Run the production loop

Historical replay must use an artifact trained strictly before the replay date.
For the checked Week 4 replay store, that artifact is
`models/replay_2026-06-14`.

```bash
# One replay-safe forecast day
make forecast AS_OF=2026-06-15 MODEL_DIR=models/replay_2026-06-14

# Serve the precomputed prediction store
make api

# Monitor the exact artifact version that produced the store
make monitor MODEL_DIR=models/replay_2026-06-14

# Evaluate champion/challenger promotion on a truly unseen full month
make promote
```

Monitoring writes `reports/monitoring_summary.json` and
`reports/drift_report.html`. Exit code `0` means both gates passed, `1` means
performance passed but drift needs review, and `2` means the performance gate
failed. Promotion returns `waiting_for_unseen_month` rather than evaluating on
data included in the champion's training cutoff.

### Docker

```bash
make docker-build
make docker-up
curl http://127.0.0.1:8000/health

docker compose run --rm monitoring \
  --model-dir models/replay_2026-06-14

docker compose run --rm promotion
make docker-down
```

The API container mounts `data/` and `models/` read-only. Forecast,
monitoring, and promotion are opt-in one-shot Compose jobs. The CI workflow
runs lint, all tests, Compose validation, and a production image build.

## Roadmap

- [x] Week 1 — Data cleaning, station mapping, scope selection, EDA
- [x] Week 2 — Station-hour grid, leak-safe features, rolling-origin backtest, baselines
- [x] Week 3 — LightGBM model, feature iteration, tuning, error analysis
- [x] Week 4 — Self-describing artifacts, leak-safe batch forecast, Prefect replay, and FastAPI serving
- [x] Week 5 — Version-matched monitoring, leakage-safe promotion, Docker Compose, and GitHub Actions
