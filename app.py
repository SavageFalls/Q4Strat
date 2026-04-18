from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html
from plotly.subplots import make_subplots

from data import load_sp500_data
from strategy import (
    QUARTER_ORDER,
    StrategyConfig,
    available_strategy_combos,
    benchmark_result,
    combo_name,
    parse_combo_name,
    run_strategy,
)


APP_TITLE = "S&P 500 Seasonal Quarter Strategy Dashboard"
DEFAULT_PORT = 8050
EXPORT_DIR = Path("exports")
EXPORT_DIR.mkdir(exist_ok=True)

DATA = load_sp500_data()
MIN_DATE = DATA.index.min().date()
MAX_DATE = DATA.index.max().date()
DEFAULT_EXCLUDED_YEARS: list[int] = []
COMBO_OPTIONS = available_strategy_combos()


def quant_style() -> dict[str, str]:
    return {
        "paper_bgcolor": "#0b1220",
        "plot_bgcolor": "#111a2c",
        "font": {"color": "#dbe4f0", "family": "Aptos, Segoe UI, sans-serif"},
        "hoverlabel": {"bgcolor": "#111a2c"},
    }


def format_pct(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2%}"


def format_num(value: float) -> str:
    return "N/A" if pd.isna(value) else f"{value:.2f}"


def serialize_parameters(params: dict[str, object]) -> None:
    latest_path = EXPORT_DIR / "latest_parameters.json"
    latest_path.write_text(json.dumps(params, indent=2), encoding="utf-8")


def metric_card(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        className="metric-card",
        children=[
            html.Div(title, className="metric-label"),
            html.Div(value, className="metric-value"),
            html.Div(subtitle, className="metric-subtitle"),
        ],
    )


def filter_data(data: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    frame = data.loc[pd.to_datetime(start_date) : pd.to_datetime(end_date)].copy()
    if frame.empty:
        raise ValueError("Selected date range returned no observations.")
    return frame


def build_results(
    data: pd.DataFrame,
    primary_quarters: Iterable[str],
    comparison_names: list[str],
    regime_filter: str,
    excluded_years: list[int],
    rolling_years: int,
) -> tuple[dict[str, object], list[tuple[str, object]]]:
    primary_name = combo_name(primary_quarters)
    strategy_names = [primary_name]
    for name in comparison_names:
        if name != primary_name:
            strategy_names.append(name)

    results: list[tuple[str, object]] = []
    benchmark = benchmark_result(
        data=data,
        rolling_window_years=rolling_years,
        excluded_years=excluded_years,
    )
    results.append(("Buy & Hold", benchmark))

    for strategy_name in strategy_names:
        config = StrategyConfig(
            name=strategy_name,
            quarters=parse_combo_name(strategy_name),
            regime_filter=regime_filter,
            excluded_years=tuple(excluded_years),
            rolling_window_years=rolling_years,
        )
        results.append((strategy_name, run_strategy(data, config)))

    primary_result = next(result for name, result in results if name == primary_name)
    return primary_result, results


def make_equity_figure(results: list[tuple[str, object]], use_log_scale: bool) -> go.Figure:
    figure = go.Figure()
    for name, result in results:
        figure.add_trace(
            go.Scatter(
                x=result.equity_curve.index,
                y=result.equity_curve,
                mode="lines",
                name=name,
                hovertemplate="%{x|%Y-%m-%d}<br>Equity: %{y:.3f}<extra>%{fullData.name}</extra>",
            )
        )

    figure.update_layout(
        title="Equity Curve",
        yaxis_title="Growth of $1",
        xaxis_title="Date",
        legend_title="Series",
        **quant_style(),
    )
    if use_log_scale:
        figure.update_yaxes(type="log")
    return figure


def make_drawdown_figure(results: list[tuple[str, object]]) -> go.Figure:
    figure = go.Figure()
    for name, result in results:
        figure.add_trace(
            go.Scatter(
                x=result.drawdown.index,
                y=result.drawdown,
                mode="lines",
                name=name,
                hovertemplate="%{x|%Y-%m-%d}<br>Drawdown: %{y:.2%}<extra>%{fullData.name}</extra>",
            )
        )

    figure.update_layout(
        title="Drawdown",
        yaxis_title="Drawdown",
        xaxis_title="Date",
        **quant_style(),
    )
    figure.update_yaxes(tickformat=".0%")
    return figure


def make_rolling_figure(results: list[tuple[str, object]]) -> go.Figure:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Rolling Sharpe", "Rolling Sortino", "Rolling Calmar", "Rolling Volatility"),
        shared_xaxes=True,
    )
    metrics = [("sharpe", 1, 1), ("sortino", 1, 2), ("calmar", 2, 1), ("volatility", 2, 2)]
    for metric_name, row, col in metrics:
        for strategy_name, result in results:
            if result.rolling.empty:
                continue
            figure.add_trace(
                go.Scatter(
                    x=result.rolling.index,
                    y=result.rolling[metric_name],
                    mode="lines",
                    name=strategy_name,
                    legendgroup=strategy_name,
                    showlegend=(metric_name == "sharpe"),
                    hovertemplate="%{x|%Y-%m-%d}<br>"
                    + f"{metric_name.title()}: "
                    + ("%{y:.2%}" if metric_name == "volatility" else "%{y:.2f}")
                    + "<extra>%{fullData.name}</extra>",
                ),
                row=row,
                col=col,
            )
    figure.update_layout(title="Rolling Risk Metrics", **quant_style())
    figure.update_yaxes(tickformat=".0%", row=2, col=2)
    return figure


def make_histogram_figure(results: list[tuple[str, object]]) -> go.Figure:
    figure = go.Figure()
    for strategy_name, result in results:
        if result.selected_periods.empty:
            continue
        figure.add_trace(
            go.Histogram(
                x=result.selected_periods["period_return"],
                name=strategy_name,
                opacity=0.6,
                nbinsx=25,
                hovertemplate="Return bin: %{x:.2%}<br>Count: %{y}<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        title="Quarter Return Distribution",
        barmode="overlay",
        xaxis_title="Quarter Return",
        yaxis_title="Count",
        **quant_style(),
    )
    figure.update_xaxes(tickformat=".0%")
    return figure


def make_heatmap_figure(primary_result) -> go.Figure:
    heatmap = primary_result.quarter_heatmap.sort_index(ascending=False)
    figure = go.Figure(
        go.Heatmap(
            z=heatmap.values,
            x=heatmap.columns,
            y=heatmap.index.astype(str),
            colorscale="RdYlGn",
            zmid=0.0,
            hovertemplate="Year %{y}<br>%{x}: %{z:.2%}<extra></extra>",
            colorbar={"tickformat": ".0%", "title": "Return"},
        )
    )
    figure.update_layout(title="Year-by-Year Quarter Return Heatmap", **quant_style())
    return figure


def make_metrics_table(results: list[tuple[str, object]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows = []
    for strategy_name, result in results:
        rows.append(
            {
                "Strategy": strategy_name,
                "Total Return": format_pct(result.metrics["total_return"]),
                "CAGR": format_pct(result.metrics["cagr"]),
                "Volatility": format_pct(result.metrics["annualized_volatility"]),
                "Sharpe": format_num(result.metrics["sharpe_ratio"]),
                "Sortino": format_num(result.metrics["sortino_ratio"]),
                "Calmar": format_num(result.metrics["calmar_ratio"]),
                "Max Drawdown": format_pct(result.metrics["max_drawdown"]),
                "Drawdown Days": str(int(result.metrics["drawdown_duration_days"])) if pd.notna(result.metrics["drawdown_duration_days"]) else "N/A",
                "Ulcer Index": format_num(result.metrics["ulcer_index"]),
                "Downside Deviation": format_pct(result.metrics["downside_deviation"]),
                "Skewness": format_num(result.metrics["skewness"]),
                "Kurtosis": format_num(result.metrics["kurtosis"]),
                "Win Rate": format_pct(result.metrics["win_rate"]),
            }
        )
    columns = [{"name": key, "id": key} for key in rows[0].keys()] if rows else []
    return rows, columns


def build_period_export(results: list[tuple[str, object]]) -> pd.DataFrame:
    exports = []
    for strategy_name, result in results:
        periods = result.selected_periods.copy()
        if periods.empty:
            continue
        periods.insert(0, "strategy", strategy_name)
        exports.append(periods)
    return pd.concat(exports, ignore_index=True) if exports else pd.DataFrame()


app: Dash = dash.Dash(__name__)
app.title = APP_TITLE

app.layout = html.Div(
    className="app-shell",
    children=[
        dcc.Download(id="download-metrics"),
        dcc.Download(id="download-periods"),
        html.Div(
            className="sidebar",
            children=[
                html.H1("Seasonal Quant Lab", className="app-title"),
                html.P("S&P 500 quarter-seasonality research dashboard using adjusted daily closes.", className="sidebar-copy"),
                html.Label("Primary Quarter Selection", className="control-label"),
                dcc.Checklist(
                    id="quarter-selection",
                    options=[{"label": quarter, "value": quarter} for quarter in QUARTER_ORDER],
                    value=["Q4"],
                    className="checklist",
                ),
                html.Label("Comparison Strategies", className="control-label"),
                dcc.Dropdown(
                    id="comparison-strategies",
                    options=[{"label": name, "value": name} for name in COMBO_OPTIONS],
                    value=["Q4+Q1"],
                    multi=True,
                    className="control",
                ),
                html.Label("Regime Filter", className="control-label"),
                dcc.Dropdown(
                    id="regime-filter",
                    options=[
                        {"label": "None", "value": "none"},
                        {"label": "Risk-On: Price > 200DMA", "value": "ma_200"},
                        {"label": "Risk-On: 21D Vol < 252D Median", "value": "vol_21_below_252_median"},
                        {"label": "Combined: 200DMA + Volatility", "value": "ma_200_and_vol"},
                    ],
                    value="none",
                    clearable=False,
                    className="control",
                ),
                html.Label("Exclude Calendar Years", className="control-label"),
                dcc.Dropdown(
                    id="excluded-years",
                    options=[{"label": str(year), "value": year} for year in sorted(DATA.index.year.unique())],
                    value=DEFAULT_EXCLUDED_YEARS,
                    multi=True,
                    className="control",
                ),
                html.Label("Rolling Window (Years)", className="control-label"),
                dcc.Slider(
                    id="rolling-window",
                    min=1,
                    max=10,
                    step=1,
                    value=5,
                    marks={year: str(year) for year in range(1, 11)},
                ),
                html.Label("Date Range", className="control-label"),
                dcc.DatePickerRange(
                    id="date-range",
                    start_date=MIN_DATE,
                    end_date=MAX_DATE,
                    min_date_allowed=MIN_DATE,
                    max_date_allowed=MAX_DATE,
                    display_format="YYYY-MM-DD",
                    className="date-range",
                ),
                html.Label("Equity Axis", className="control-label"),
                dcc.RadioItems(
                    id="equity-scale",
                    options=[
                        {"label": "Linear", "value": "linear"},
                        {"label": "Log", "value": "log"},
                    ],
                    value="linear",
                    className="radio-group",
                ),
                html.Div(
                    className="button-row",
                    children=[
                        html.Button("Export Metrics CSV", id="export-metrics", n_clicks=0, className="action-button"),
                        html.Button("Export Period Returns CSV", id="export-periods", n_clicks=0, className="action-button"),
                    ],
                ),
                html.Div(
                    className="sidebar-footnote",
                    children=f"Loaded data: {MIN_DATE} to {MAX_DATE}",
                ),
            ],
        ),
        html.Div(
            className="main-panel",
            children=[
                html.Div(id="kpi-row", className="kpi-grid"),
                html.Div(
                    className="table-card",
                    children=[
                        html.H2("Strategy Comparison", className="section-title"),
                        dash_table.DataTable(
                            id="metrics-table",
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "backgroundColor": "#111a2c",
                                "color": "#dbe4f0",
                                "border": "1px solid #1f2c45",
                                "padding": "8px",
                                "fontFamily": "Aptos, Segoe UI, sans-serif",
                            },
                            style_header={
                                "backgroundColor": "#18253c",
                                "fontWeight": "bold",
                                "color": "#f8fafc",
                                "border": "1px solid #2b3b59",
                            },
                        ),
                    ],
                ),
                dcc.Graph(id="equity-curve"),
                dcc.Graph(id="drawdown-chart"),
                dcc.Graph(id="rolling-panel"),
                dcc.Graph(id="return-distribution"),
                dcc.Graph(id="quarter-heatmap"),
            ],
        ),
    ],
)


@callback(
    Output("kpi-row", "children"),
    Output("metrics-table", "data"),
    Output("metrics-table", "columns"),
    Output("equity-curve", "figure"),
    Output("drawdown-chart", "figure"),
    Output("rolling-panel", "figure"),
    Output("return-distribution", "figure"),
    Output("quarter-heatmap", "figure"),
    Input("quarter-selection", "value"),
    Input("comparison-strategies", "value"),
    Input("regime-filter", "value"),
    Input("excluded-years", "value"),
    Input("rolling-window", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("equity-scale", "value"),
)
def update_dashboard(
    quarter_selection: list[str],
    comparison_strategies: list[str],
    regime_filter: str,
    excluded_years: list[int],
    rolling_window: int,
    start_date: str,
    end_date: str,
    equity_scale: str,
):
    quarter_selection = quarter_selection or ["Q4"]
    filtered = filter_data(DATA, start_date, end_date)
    primary_result, results = build_results(
        data=filtered,
        primary_quarters=quarter_selection,
        comparison_names=comparison_strategies or [],
        regime_filter=regime_filter,
        excluded_years=excluded_years or [],
        rolling_years=rolling_window,
    )

    serialize_parameters(
        {
            "primary_quarters": quarter_selection,
            "comparison_strategies": comparison_strategies or [],
            "regime_filter": regime_filter,
            "excluded_years": excluded_years or [],
            "rolling_window_years": rolling_window,
            "start_date": start_date,
            "end_date": end_date,
            "equity_scale": equity_scale,
        }
    )

    metrics = primary_result.metrics
    kpis = [
        metric_card("Primary Strategy", primary_result.config.name, "Seasonal quarter blend"),
        metric_card("CAGR", format_pct(metrics["cagr"])),
        metric_card("Total Return", format_pct(metrics["total_return"])),
        metric_card("Sharpe Ratio", format_num(metrics["sharpe_ratio"])),
        metric_card("Sortino Ratio", format_num(metrics["sortino_ratio"])),
        metric_card("Max Drawdown", format_pct(metrics["max_drawdown"])),
        metric_card("Win Rate", format_pct(metrics["win_rate"])),
        metric_card("Ulcer Index", format_num(metrics["ulcer_index"])),
    ]
    table_data, table_columns = make_metrics_table(results)
    return (
        kpis,
        table_data,
        table_columns,
        make_equity_figure(results, use_log_scale=(equity_scale == "log")),
        make_drawdown_figure(results),
        make_rolling_figure(results),
        make_histogram_figure(results),
        make_heatmap_figure(primary_result),
    )


@callback(
    Output("download-metrics", "data"),
    Input("export-metrics", "n_clicks"),
    State("quarter-selection", "value"),
    State("comparison-strategies", "value"),
    State("regime-filter", "value"),
    State("excluded-years", "value"),
    State("rolling-window", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    prevent_initial_call=True,
)
def export_metrics(
    _: int,
    quarter_selection: list[str],
    comparison_strategies: list[str],
    regime_filter: str,
    excluded_years: list[int],
    rolling_window: int,
    start_date: str,
    end_date: str,
):
    filtered = filter_data(DATA, start_date, end_date)
    _, results = build_results(
        data=filtered,
        primary_quarters=quarter_selection or ["Q4"],
        comparison_names=comparison_strategies or [],
        regime_filter=regime_filter,
        excluded_years=excluded_years or [],
        rolling_years=rolling_window,
    )
    rows, _ = make_metrics_table(results)
    export_frame = pd.DataFrame(rows)
    return dcc.send_data_frame(export_frame.to_csv, "strategy_metrics.csv", index=False)


@callback(
    Output("download-periods", "data"),
    Input("export-periods", "n_clicks"),
    State("quarter-selection", "value"),
    State("comparison-strategies", "value"),
    State("regime-filter", "value"),
    State("excluded-years", "value"),
    State("rolling-window", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
    prevent_initial_call=True,
)
def export_periods(
    _: int,
    quarter_selection: list[str],
    comparison_strategies: list[str],
    regime_filter: str,
    excluded_years: list[int],
    rolling_window: int,
    start_date: str,
    end_date: str,
):
    filtered = filter_data(DATA, start_date, end_date)
    _, results = build_results(
        data=filtered,
        primary_quarters=quarter_selection or ["Q4"],
        comparison_names=comparison_strategies or [],
        regime_filter=regime_filter,
        excluded_years=excluded_years or [],
        rolling_years=rolling_window,
    )
    export_frame = build_period_export(results)
    return dcc.send_data_frame(export_frame.to_csv, "period_returns.csv", index=False)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=DEFAULT_PORT, debug=False)
