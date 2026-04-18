"""Strategy engine for quarter-seasonality deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


QUARTER_MONTHS = {
    "Q1": {1, 2, 3},
    "Q2": {4, 5, 6},
    "Q3": {7, 8, 9},
    "Q4": {10, 11, 12},
}


@dataclass(frozen=True)
class RegimeConfig:
    enabled: bool = False
    mode: str = "risk_on"  # risk_on | risk_off
    fast_ma: int = 50
    slow_ma: int = 200


def _quarter_mask(index: pd.DatetimeIndex, selected_quarters: Iterable[str]) -> pd.Series:
    selected = set(selected_quarters)
    if not selected:
        return pd.Series(False, index=index)

    valid = selected.intersection(set(QUARTER_MONTHS))
    months = set().union(*(QUARTER_MONTHS[q] for q in valid)) if valid else set()
    return pd.Series(index.month.isin(months), index=index)


def _regime_mask(prices: pd.Series, config: RegimeConfig) -> pd.Series:
    if not config.enabled:
        return pd.Series(True, index=prices.index)

    fast = prices.rolling(config.fast_ma, min_periods=config.fast_ma).mean()
    slow = prices.rolling(config.slow_ma, min_periods=config.slow_ma).mean()

    risk_on = fast > slow
    if config.mode == "risk_off":
        mask = ~risk_on
    else:
        mask = risk_on

    return mask.fillna(False)


def compute_strategy_returns(
    prices: pd.DataFrame,
    selected_quarters: Iterable[str],
    regime_config: Optional[RegimeConfig] = None,
) -> pd.DataFrame:
    """Compute daily strategy and benchmark returns.

    Strategy is in market only when current date falls in a selected quarter
    and (optionally) regime condition is satisfied.
    """

    regime_config = regime_config or RegimeConfig(enabled=False)

    df = prices.copy()
    if "daily_return" not in df:
        df["daily_return"] = df["adj_close"].pct_change().fillna(0.0)

    quarter_mask = _quarter_mask(df.index, selected_quarters)
    regime_mask = _regime_mask(df["adj_close"], regime_config)

    df["in_market"] = (quarter_mask & regime_mask).astype(int)
    df["strategy_return"] = np.where(df["in_market"] == 1, df["daily_return"], 0.0)
    df["benchmark_return"] = df["daily_return"]

    # Equity curves start at 1.0 for straightforward comparability.
    df["strategy_equity"] = (1.0 + df["strategy_return"]).cumprod()
    df["benchmark_equity"] = (1.0 + df["benchmark_return"]).cumprod()

    return df


def period_returns(
    strategy_df: pd.DataFrame,
    selected_quarters: Iterable[str],
) -> pd.DataFrame:
    """Compute quarter period returns for selected quarters only."""

    selected = set(selected_quarters)

    period = strategy_df.copy()
    period["year"] = period.index.year
    period["quarter_num"] = period.index.quarter
    period["quarter_label"] = "Q" + period["quarter_num"].astype(str)

    period = period[period["quarter_label"].isin(selected)]

    grouped = period.groupby(["year", "quarter_label"], sort=True)
    out = grouped["strategy_return"].apply(lambda x: (1.0 + x).prod() - 1.0)
    out = out.reset_index(name="period_return")

    return out


def maybe_exclude_years(df: pd.DataFrame, excluded_years: Iterable[int]) -> pd.DataFrame:
    excluded = set(int(y) for y in excluded_years)
    if not excluded:
        return df
    return df[~df.index.year.isin(excluded)].copy()
