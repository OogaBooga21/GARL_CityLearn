# EC_RL Dashboard — Feature List

## Platform Overview

Built with vanilla JavaScript + Plotly.js (v2.35.2) served via a lightweight Flask backend. No heavy ML framework required to view results. All data flows **automatically** from simulation → KPI calculation → JSON index → web UI — zero manual file uploading.

---

## KPIs & Metrics Tracked

### Per-building time-series (one trace per building, e.g. 9 buildings)

| KPI | Unit |
|-----|------|
| Net energy exchange (grid imports/exports) | kWh |
| Energy exported to grid | kWh |
| Energy imported from grid | kWh |
| Load (non-shiftable + battery charging) | kWh |
| Non-shiftable load | kWh |
| Electricity cost | $ |
| PV generation | kWh |
| Battery state of charge (SoC) | % |
| Battery action (charge/discharge command) | Action |
| Battery discharge rate | kW |

### Aggregate time-series (summed across all buildings)

- `total_*` equivalents of all above; SoC uses capacity-weighted average

### Scalar summary KPIs (whole-run)

- Total cost, total carbon emissions, peak grid consumption, peak building load, total PV generation, average SoC, per-building charge/discharge totals

---

## UI & Dashboard Features

### Panels & Plots

- **Multiple panels** — "Add View" creates unlimited independent panels, each closeable
- **Multiple plots per panel** — overlay as many metrics and runs as desired on a single shared axis
- **Multiple options per plot trace:**
  - **Color** — HTML5 color picker; user chooses base color
  - **Line style** — solid / dotted / dashed (accessibility for colorblind users)
  - **Automatic shade generation** — when a per-building metric is added, all N buildings get mathematically consistent shades of the chosen color (8% darkening per building), keeping one metric visually grouped without manual configuration

### Legend with Inline Statistics

Each trace in the legend shows, computed over the full time series:

- **Min**, **Max**, **Avg** — always from the raw (unsmoothed) data, so stats are ground-truth regardless of smoothing setting
- Compact number formatting: 2 d.p. for normal ranges, scientific notation for |n| ≥ 10,000

### Smoothing

- **Global EMA smoothing slider** (0–0.99) — applies instantly to all panels and traces simultaneously
- Algorithm: TensorFlow-style exponential moving average (`y_smooth = w·y_prev + (1−w)·y_raw`)
- **Non-destructive**: raw data preserved in memory; only the rendered trace changes
- **Range-preserving**: the user's current zoom/pan is maintained when adjusting the smoothing weight

### Time-Series Navigation

- **Plotly range slider** on every panel's x-axis — drag to zoom into any time window
- Zoomed range survives smoothing adjustments (explicitly restored after re-render)

### Run Comparison

- Add metrics from **different simulation runs** into the same panel for direct overlaid comparison
- Or place runs in separate panels for side-by-side structural comparison
- Each trace is labelled `{run} — {building}` with its own color, line style, and inline stats

### Data Pipeline (No Manual Upload)

1. `main.py` runs the agent and writes CityLearn output CSVs
2. `kpi_calculator.py` processes outputs → structured KPI CSVs in `calculated_kpis/{run_id}/`
3. `plot_kpis.py` scans the directory and writes `available_data.json` (the dashboard index)
4. The web UI reads the index on load; fetches data on demand via `/api/kpi/<run>/<kpi>`

No restart, no upload step — open the dashboard, new runs appear automatically.

---

## Comparison with TensorBoard and CityLearn UI

| Feature | **This Dashboard** | **TensorBoard** | **CityLearn UI** |
|---|---|---|---|
| **Domain** | Multi-building energy RL | Generic ML training | CityLearn env only |
| **No manual upload** | Yes (file-watch pipeline) | No (tfevents must be written & dir pointed to) | No |
| **Multi-panel** | Yes, unlimited, dynamic | Yes (sidebar-driven, fixed layout) | Limited |
| **Multi-trace per panel** | Yes, any run × any metric | Yes | No |
| **Per-agent/building breakdown** | Yes, auto-shaded | No | Yes |
| **Min/Max/Avg in legend** | Yes, always visible | No | No |
| **EMA smoothing slider** | Yes (global, real-time) | Yes (per-section) | No |
| **Color picker per trace** | Yes | No (auto-assigned) | No |
| **Line style (solid/dot/dash)** | Yes | No | No |
| **Range slider (zoom)** | Yes (Plotly native) | Yes | No |
| **Domain-specific KPIs** | Yes (SoC, cost, carbon, PV, etc.) | No (loss/accuracy paradigm) | Yes |
| **Automatic unit labelling** | Yes | No | Partial |
| **Lightweight (no TF dependency)** | Yes (Flask + Plotly.js) | No (requires TensorFlow) | Depends on CityLearn install |
| **Run comparison** | Yes (same panel or side-by-side) | Yes (overlay experiments) | No |

---

## Additional Nice-to-Haves

- **Automatic shade cascade** — adding a 9-building metric never requires 9 color selections; one base color fans out automatically
- **Metric tokens/tags** — each active trace shown as a labelled pill below the plot with a one-click remove button, so the panel's composition is always visible at a glance
- **Y-axis auto-unit** — axis label is set from the KPI's unit metadata automatically on the first trace added
- **Compact number formatting** in legend stats (scientific notation kicks in for large values like total kWh across a year)
- **Modal-based trace addition** — clean, non-destructive workflow: pick run → pick metric → pick color → pick line style → add; the plot re-renders without losing anything else
- **Escape key closes modal** — keyboard UX shortcut
- **`preserveRange` re-render** — panning/zooming into an interesting time window and then adjusting smoothing does not snap back to the full view
- **Weighted SoC aggregate** — `total_electrical_storage_soc` is computed as a battery-capacity-weighted mean, not a naive average, which is the physically correct aggregation for SoC
