from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd

from metrics import equity_curve_from_returns, rolling_metrics, summarize_strategy


QUARTER_ORDER = ("Q1", "Q2", "Q3", "Q4")
ROLLING_TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    quarters: tuple[str, ...]
    regime_filter: str = "none"
    excluded_years: tuple[int, ...] = ()
    rolling_window_years: int = 5


@dataclass
class StrategyResult:
    config: StrategyConfig
    signal: pd.Series
    regime_mask: pd.Series
    daily_returns: pd.Series
    equity_curve: pd.Series
    drawdown: pd.Series
    selected_periods: pd.DataFrame
    quarter_heatmap: pd.DataFrame
    yearly_selected_returns: pd.Series
    metrics: dict[str, float]
    rolling: pd.DataFrame


def canonical_quarter_combo(quarters: Sequence[str]) -> tuple[str, ...]:
    quarter_set = set(quarters)
    return tuple(quarter for quarter in QUARTER_ORDER if quarter in quarter_set)


def combo_name(quarters: Sequence[str]) -> str:
    ordered = canonical_quarter_combo(quarters)
    return "+".join(ordered) if ordered else "No Quarter"


def available_strategy_combos() -> list[str]:
    combos: list[str] = []
    for size in range(1, len(QUARTER_ORDER) + 1):
        combos.extend(combo_name(combo) for combo in combinations(QUARTER_ORDER, size))
    return combos


def parse_combo_name(name: str) -> tuple[str, ...]:
    return canonical_quarter_combo(name.split("+"))


def regime_mask(data: pd.DataFrame, regime: str) -> pd.Series:
    price = data["adj_close"]
    if regime == "none":
        return pd.Series(True, index=data.index)

    ma200 = price.rolling(200, min_periods=200).mean()
    short_vol = data["daily_return"].rolling(21, min_periods=21).std(ddof=1) * np.sqrt(252)
    vol_threshold = short_vol.rolling(252, min_periods=126).median()

    ma_signal = price > ma200
    vol_signal = short_vol < vol_threshold

    if regime == "ma_200":
        return ma_signal.fillna(False)
    if regime == "vol_21_below_252_median":
        return vol_signal.fillna(False)
    if regime == "ma_200_and_vol":
        return (ma_signal & vol_signal).fillna(False)

    raise ValueError(f"Unsupported regime filter: {regime}")


def build_strategy_signal(
    data: pd.DataFrame,
    quarters: Sequence[str],
    regime: str = "none",
    excluded_years: Sequence[int] = (),
) -> tuple[pd.Series, pd.Series]:
    selected_quarters = canonical_quarter_combo(quarters)
    if not selected_quarters:
        signal = pd.Series(False, index=data.index)
        return signal, pd.Series(True, index=data.index)

    quarter_mask = data["quarter"].isin(selected_quarters)
    year_mask = ~data.index.year.isin(list(excluded_years))
    regime_series = regime_mask(data, regime)
    signal = (quarter_mask & year_mask & regime_series).astype(bool)
    return signal, regime_series


def compute_selected_periods(
    data: pd.DataFrame,
    daily_returns: pd.Series,
    selected_quarters: Sequence[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "year": data.index.year,
            "quarter": data["quarter"],
            "daily_return": daily_returns,
        },
        index=data.index,
    )
    frame = frame[frame["quarter"].isin(selected_quarters)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["year", "quarter", "period_return"])

    period_returns = (
        frame.groupby(["year", "quarter"], observed=True)["daily_return"]
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        .reset_index(name="period_return")
    )
    return period_returns


def compute_quarter_heatmap(data: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "year": data.index.year,
            "quarter": data["quarter"],
            "daily_return": data["daily_return"],
        },
        index=data.index,
    )
    quarter_returns = (
        frame.groupby(["year", "quarter"], observed=True)["daily_return"]
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        .unstack("quarter")
        .reindex(columns=QUARTER_ORDER)
    )
    return quarter_returns


def run_strategy(data: pd.DataFrame, config: StrategyConfig) -> StrategyResult:
    signal, regime_series = build_strategy_signal(
        data=data,
        quarters=config.quarters,
        regime=config.regime_filter,
        excluded_years=config.excluded_years,
    )
    strategy_daily_returns = data["daily_return"].where(signal, 0.0)
    equity_curve = equity_curve_from_returns(strategy_daily_returns)
    drawdown = equity_curve.div(equity_curve.cummax()).sub(1.0)
    selected_periods = compute_selected_periods(data, strategy_daily_returns, config.quarters)
    period_series = selected_periods["period_return"] if not selected_periods.empty else pd.Series(dtype=float)
    heatmap = compute_quarter_heatmap(data.loc[~data.index.year.isin(config.excluded_years)])
    yearly_selected_returns = (
        selected_periods.groupby("year", observed=True)["period_return"]
        .apply(lambda x: (1.0 + x).prod() - 1.0)
        if not selected_periods.empty
        else pd.Series(dtype=float)
    )
    metrics = summarize_strategy(strategy_daily_returns, period_series, equity_curve)
    rolling = rolling_metrics(
        strategy_daily_returns,
        window_days=config.rolling_window_years * ROLLING_TRADING_DAYS_PER_YEAR,
    )

    return StrategyResult(
        config=config,
        signal=signal,
        regime_mask=regime_series,
        daily_returns=strategy_daily_returns,
        equity_curve=equity_curve,
        drawdown=drawdown,
        selected_periods=selected_periods,
        quarter_heatmap=heatmap,
        yearly_selected_returns=yearly_selected_returns,
        metrics=metrics,
        rolling=rolling,
    )


def benchmark_result(data: pd.DataFrame, rolling_window_years: int, excluded_years: Sequence[int] = ()) -> StrategyResult:
    year_mask = ~data.index.year.isin(list(excluded_years))
    benchmark_returns = data["daily_return"].where(year_mask, 0.0)
    equity_curve = equity_curve_from_returns(benchmark_returns)
    drawdown = equity_curve.div(equity_curve.cummax()).sub(1.0)
    heatmap = compute_quarter_heatmap(data.loc[year_mask])
    selected_periods = compute_selected_periods(data.loc[year_mask], benchmark_returns.loc[year_mask], QUARTER_ORDER)
    metrics = summarize_strategy(
        benchmark_returns,
        selected_periods["period_return"] if not selected_periods.empty else pd.Series(dtype=float),
        equity_curve,
    )
    rolling = rolling_metrics(
        benchmark_returns,
        window_days=rolling_window_years * ROLLING_TRADING_DAYS_PER_YEAR,
    )
    return StrategyResult(
        config=StrategyConfig(name="Buy & Hold", quarters=QUARTER_ORDER, excluded_years=tuple(excluded_years)),
        signal=pd.Series(True, index=data.index),
        regime_mask=pd.Series(True, index=data.index),
        daily_returns=benchmark_returns,
        equity_curve=equity_curve,
        drawdown=drawdown,
        selected_periods=selected_periods,
        quarter_heatmap=heatmap,
        yearly_selected_returns=(
            selected_periods.groupby("year", observed=True)["period_return"]
            .apply(lambda x: (1.0 + x).prod() - 1.0)
            if not selected_periods.empty
            else pd.Series(dtype=float)
        ),
        metrics=metrics,
        rolling=rolling,
    )
