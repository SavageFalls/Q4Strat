"""Streamlit dashboard for S&P 500 seasonal quarter strategy research."""

from __future__ import annotations

import io
import json
from datetime import date

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import get_sp500_daily
from metrics import compute_performance_metrics, drawdown_series, rolling_metrics
from strategy import RegimeConfig, compute_strategy_returns, maybe_exclude_years, period_returns

st.set_page_config(page_title="S&P 500 Seasonal Strategy Lab", layout="wide")

st.title("S&P 500 Seasonal Equity Strategy Lab")
st.caption("Institutional-grade quarter seasonality research engine (^GSPC, daily, adjusted close)")


@st.cache_data(show_spinner=False)
def load_data(start: str, end: str) -> pd.DataFrame:
    return get_sp500_daily(start=start, end=end)


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.2%}"


def fmt_num(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:.2f}"


with st.sidebar:
    st.header("Controls")

    quarter_choices = ["Q1", "Q2", "Q3", "Q4"]
    selected_quarters = st.multiselect(
        "Select quarters",
        options=quarter_choices,
        default=["Q4"],
    )

    rolling_window = st.slider("Rolling window (years)", min_value=2, max_value=15, value=5, step=1)

    min_date = date(2000, 1, 1)
    max_date = pd.Timestamp.now(tz="UTC").date()
    start_date, end_date = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    st.subheader("Advanced features")
    compare_q4q1 = st.checkbox("Compare vs Q4+Q1", value=True)
    log_scale = st.checkbox("Log scale equity curves", value=False)

    use_regime = st.checkbox("Enable MA regime filter", value=False)
    regime_mode = st.selectbox("Regime mode", options=["risk_on", "risk_off"], index=0)
    fast_ma = st.number_input("Fast MA", min_value=10, max_value=200, value=50, step=5)
    slow_ma = st.number_input("Slow MA", min_value=50, max_value=400, value=200, step=10)

    exclude_2008 = st.checkbox("Exclude 2008 outlier year", value=False)

if not selected_quarters:
    st.warning("Please select at least one quarter to run strategy.")
    st.stop()

if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

with st.spinner("Loading market data..."):
    try:
        prices = load_data(str(start_date), str(end_date))
    except Exception as exc:
        st.error("Failed to load market data from provider. This may be temporary (e.g., API rate limits).")
        st.exception(exc)
        st.stop()


if exclude_2008:
    prices = maybe_exclude_years(prices, [2008])

regime_cfg = RegimeConfig(
    enabled=use_regime,
    mode=regime_mode,
    fast_ma=int(fast_ma),
    slow_ma=int(slow_ma),
)

strategy_df = compute_strategy_returns(prices, selected_quarters=selected_quarters, regime_config=regime_cfg)

main_metrics = compute_performance_metrics(strategy_df["strategy_return"], strategy_df["strategy_equity"])
bench_metrics = compute_performance_metrics(strategy_df["benchmark_return"], strategy_df["benchmark_equity"])

dd_strategy = drawdown_series(strategy_df["strategy_equity"])
dd_benchmark = drawdown_series(strategy_df["benchmark_equity"])

roll = rolling_metrics(
    returns=strategy_df["strategy_return"],
    equity=strategy_df["strategy_equity"],
    window_years=rolling_window,
)

period_df = period_returns(strategy_df, selected_quarters=selected_quarters)

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Strategy CAGR", fmt_pct(main_metrics["cagr"]))
k2.metric("Total Return", fmt_pct(main_metrics["total_return"]))
k3.metric("Ann. Volatility", fmt_pct(main_metrics["annualized_volatility"]))
k4.metric("Sharpe", fmt_num(main_metrics["sharpe"]))
k5.metric("Sortino", fmt_num(main_metrics["sortino"]))
k6.metric("Calmar", fmt_num(main_metrics["calmar"]))

k7, k8, k9, k10, k11 = st.columns(5)
k7.metric("Max Drawdown", fmt_pct(main_metrics["max_drawdown"]))
k8.metric("Drawdown Duration (days)", str(main_metrics["drawdown_duration"]))
k9.metric("Ulcer Index", fmt_num(main_metrics["ulcer_index"]))
k10.metric("Downside Deviation", fmt_pct(main_metrics["downside_deviation"]))
k11.metric("Win Rate", fmt_pct(main_metrics["win_rate"]))

st.markdown("---")

fig_equity = go.Figure()
fig_equity.add_trace(
    go.Scatter(
        x=strategy_df.index,
        y=strategy_df["strategy_equity"],
        mode="lines",
        name="Strategy",
        hovertemplate="%{x|%Y-%m-%d}<br>Equity: %{y:.4f}<extra></extra>",
    )
)
fig_equity.add_trace(
    go.Scatter(
        x=strategy_df.index,
        y=strategy_df["benchmark_equity"],
        mode="lines",
        name="Buy & Hold",
        hovertemplate="%{x|%Y-%m-%d}<br>Equity: %{y:.4f}<extra></extra>",
    )
)

if compare_q4q1:
    compare_df = compute_strategy_returns(prices, selected_quarters=["Q4", "Q1"], regime_config=regime_cfg)
    fig_equity.add_trace(
        go.Scatter(
            x=compare_df.index,
            y=compare_df["strategy_equity"],
            mode="lines",
            name="Comparator (Q4+Q1)",
            hovertemplate="%{x|%Y-%m-%d}<br>Equity: %{y:.4f}<extra></extra>",
        )
    )

fig_equity.update_layout(
    title="Equity Curve: Strategy vs Benchmark",
    yaxis_title="Growth of $1",
    xaxis_title="Date",
    hovermode="x unified",
)
if log_scale:
    fig_equity.update_yaxes(type="log")
st.plotly_chart(fig_equity, use_container_width=True)

fig_dd = go.Figure()
fig_dd.add_trace(
    go.Scatter(x=strategy_df.index, y=dd_strategy, mode="lines", name="Strategy DD")
)
fig_dd.add_trace(
    go.Scatter(x=strategy_df.index, y=dd_benchmark, mode="lines", name="Benchmark DD")
)
fig_dd.update_layout(
    title="Drawdown Comparison",
    yaxis_title="Drawdown",
    xaxis_title="Date",
    hovermode="x unified",
)
fig_dd.update_yaxes(tickformat=".1%")
st.plotly_chart(fig_dd, use_container_width=True)

roll_plot = roll.dropna().reset_index().rename(columns={"index": "date"})
if not roll_plot.empty:
    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=roll_plot["date"], y=roll_plot["rolling_sharpe"], name="Rolling Sharpe"))
    fig_roll.add_trace(go.Scatter(x=roll_plot["date"], y=roll_plot["rolling_sortino"], name="Rolling Sortino"))
    fig_roll.add_trace(go.Scatter(x=roll_plot["date"], y=roll_plot["rolling_calmar"], name="Rolling Calmar"))
    fig_roll.add_trace(go.Scatter(x=roll_plot["date"], y=roll_plot["rolling_volatility"], name="Rolling Volatility"))
    fig_roll.update_layout(
        title=f"Rolling Metrics ({rolling_window}Y window)",
        xaxis_title="Date",
        yaxis_title="Metric",
        hovermode="x unified",
    )
    st.plotly_chart(fig_roll, use_container_width=True)
else:
    st.info("Rolling window too large for selected date range.")

hist_source = period_df["period_return"] if not period_df.empty else pd.Series(dtype=float)
if len(hist_source):
    fig_hist = px.histogram(
        hist_source,
        nbins=30,
        title="Distribution of Quarter Returns",
        labels={"value": "Period Return"},
    )
    fig_hist.update_xaxes(tickformat=".1%")
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info("No period returns available for selected configuration.")

if not period_df.empty:
    pivot = period_df.pivot(index="year", columns="quarter_label", values="period_return")
    fig_heat = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="RdYlGn",
        labels=dict(x="Quarter", y="Year", color="Return"),
        title="Year-by-Year Quarter Return Heatmap",
        text_auto=".1%",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.subheader("Distribution Statistics")
stat_cols = st.columns(3)
stat_cols[0].metric("Skewness", fmt_num(main_metrics["skewness"]))
stat_cols[1].metric("Kurtosis", fmt_num(main_metrics["kurtosis"]))
stat_cols[2].metric("Win Rate (period)", fmt_pct((period_df["period_return"] > 0).mean() if len(period_df) else np.nan))

st.subheader("Strategy Configuration")
config = {
    "quarters": selected_quarters,
    "rolling_window_years": rolling_window,
    "date_start": str(start_date),
    "date_end": str(end_date),
    "regime": {
        "enabled": use_regime,
        "mode": regime_mode,
        "fast_ma": int(fast_ma),
        "slow_ma": int(slow_ma),
    },
    "exclude_2008": exclude_2008,
    "log_scale": log_scale,
    "compare_q4q1": compare_q4q1,
}
st.code(json.dumps(config, indent=2), language="json")

st.subheader("Export")
export_df = strategy_df[["strategy_return", "benchmark_return", "strategy_equity", "benchmark_equity", "in_market"]].copy()
export_df.index.name = "date"

csv_buf = io.StringIO()
export_df.to_csv(csv_buf)
st.download_button(
    "Download strategy time series (CSV)",
    data=csv_buf.getvalue(),
    file_name="seasonal_strategy_timeseries.csv",
    mime="text/csv",
)

summary_export = pd.DataFrame(
    {
        "metric": list(main_metrics.keys()),
        "strategy": list(main_metrics.values()),
        "benchmark": [bench_metrics.get(k, np.nan) for k in main_metrics.keys()],
    }
)
csv_sum = io.StringIO()
summary_export.to_csv(csv_sum, index=False)
st.download_button(
    "Download metrics summary (CSV)",
    data=csv_sum.getvalue(),
    file_name="seasonal_strategy_metrics.csv",
    mime="text/csv",
)
