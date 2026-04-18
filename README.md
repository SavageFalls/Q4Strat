# S&P 500 Seasonal Quarter Dashboard

Interactive Dash + Plotly research app for seasonal S&P 500 quarter strategies using adjusted daily closes from `^GSPC`.

## Run locally

From this folder:

```powershell
C:\Users\Lincoln\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe app.py
```

If your own Python environment already has the dependencies installed, this also works:

```powershell
python app.py
```

Then open:

[http://127.0.0.1:8050](http://127.0.0.1:8050)

## Controls

- Select any quarter combination for the primary strategy.
- Compare the primary strategy against additional quarter blends.
- Adjust the rolling window, date range, regime filter, and excluded years.
- Toggle equity scaling between linear and log.
- Export summary metrics and quarter-period returns to CSV.

## Project structure

- `data.py`: provider abstraction, Yahoo Finance loader, caching, cleaning
- `strategy.py`: quarter selection, regime filtering, period construction, benchmark logic
- `metrics.py`: institutional performance and risk calculations
- `app.py`: Dash app, callbacks, Plotly visualizations
