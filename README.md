# Divvy Demand Forecasting

Dự báo số lượt mượn xe theo trạm-giờ cho 24h tới, phục vụ bài toán rebalancing
của hệ thống bike-share Divvy (Chicago).

## Design decisions

1. Cleaning rules 60s / 24h / missing station — with the 19,74% removed per rule (section 3–4).
2. Station mapping: 1,659 IDs remapped, ambiguous pairs deliberately left unmapped (section 5).
3. Scope: `top_n_stations = 200` covering 65% of last-12-month volume (section 8) — update `config.yaml` now.
4. Zero-inflation: 52% of station-hours are zero in scope — the evidence for the Poisson objective in week 3.
5. Timezone sanity + weather join ≥95% both asserted in-code (sections 6–7).