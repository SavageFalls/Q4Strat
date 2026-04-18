from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


TRADING_DAYS = 252
CALENDAR_DAYS = 365.2425


def _as_series(values: pd.Series | Iterable[float]) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.astype(float)
    return pd.Series(values, dtype=float)


def annualized_return(daily_returns: pd.Series) -> float:
    daily_returns = _as_series(daily_returns).dropna()
    if daily_returns.empty:
        return np.nan
    return daily_returns.mean() * TRADING_DAYS


def annualized_volatility(daily_returns: pd.Series) -> float:
    daily_returns = _as_series(daily_returns).dropna()
    if len(daily_returns) < 2:
        return np.nan
    return daily_returns.std(ddof=1) * math.sqrt(TRADING_DAYS)


def downside_deviation(daily_returns: pd.Series, target: float = 0.0) -> float:
    daily_returns = _as_series(daily_returns).dropna()
    if daily_returns.empty:
        return np.nan
    downside = np.minimum(daily_returns - target, 0.0)
    return math.sqrt(np.mean(np.square(downside))) * math.sqrt(TRADING_DAYS)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    vol = annualized_volatility(daily_returns)
    if not np.isfinite(vol) or vol == 0:
        return np.nan
    excess_return = annualized_return(daily_returns) - risk_free_rate
    return excess_return / vol


def sortino_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    dd = downside_deviation(daily_returns)
    if not np.isfinite(dd) or dd == 0:
        return np.nan
    excess_return = annualized_return(daily_returns) - risk_free_rate
    return excess_return / dd


def equity_curve_from_returns(daily_returns: pd.Series, base: float = 1.0) -> pd.Series:
    daily_returns = _as_series(daily_returns).fillna(0.0)
    return base * (1.0 + daily_returns).cumprod()


def drawdown_series(equity_curve: pd.Series) -> pd.Series:
    equity_curve = _as_series(equity_curve).dropna()
    running_peak = equity_curve.cummax()
    return (equity_curve / running_peak) - 1.0


def max_drawdown(equity_curve: pd.Series) -> float:
    drawdowns = drawdown_series(equity_curve)
    if drawdowns.empty:
        return np.nan
    return drawdowns.min()


def drawdown_duration(drawdowns: pd.Series) -> int:
    drawdowns = _as_series(drawdowns).fillna(0.0)
    durations = np.where(drawdowns < 0, 1, 0)
    running = 0
    max_running = 0
    for flag in durations:
        if flag:
            running += 1
            max_running = max(max_running, running)
        else:
            running = 0
    return int(max_running)


def ulcer_index(drawdowns: pd.Series) -> float:
    drawdowns = _as_series(drawdowns).dropna()
    if drawdowns.empty:
        return np.nan
    return math.sqrt(np.mean(np.square(drawdowns * 100.0)))


def cagr(equity_curve: pd.Series) -> float:
    equity_curve = _as_series(equity_curve).dropna()
    if equity_curve.empty or len(equity_curve) < 2:
        return np.nan
    total_days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if total_days <= 0 or equity_curve.iloc[0] <= 0 or equity_curve.iloc[-1] <= 0:
        return np.nan
    years = total_days / CALENDAR_DAYS
    return (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1.0 / years) - 1.0


def calmar_ratio(equity_curve: pd.Series) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if not np.isfinite(mdd) or mdd == 0:
        return np.nan
    growth = cagr(equity_curve)
    if not np.isfinite(growth):
        return np.nan
    return growth / mdd


def period_distribution_stats(period_returns: pd.Series) -> dict[str, float]:
    period_returns = _as_series(period_returns).dropna()
    if period_returns.empty:
        return {"skewness": np.nan, "kurtosis": np.nan, "win_rate": np.nan}
    return {
        "skewness": float(skew(period_returns, bias=False)),
        "kurtosis": float(kurtosis(period_returns, fisher=True, bias=False)),
        "win_rate": float((period_returns > 0).mean()),
    }


def summarize_strategy(
    daily_returns: pd.Series,
    period_returns: pd.Series,
    equity_curve: pd.Series,
) -> dict[str, float]:
    daily_returns = _as_series(daily_returns).fillna(0.0)
    period_returns = _as_series(period_returns)
    equity_curve = _as_series(equity_curve)
    drawdowns = drawdown_series(equity_curve)
    distribution = period_distribution_stats(period_returns)

    return {
        "total_return": float(equity_curve.iloc[-1] - 1.0) if not equity_curve.empty else np.nan,
        "cagr": float(cagr(equity_curve)),
        "annualized_volatility": float(annualized_volatility(daily_returns)),
        "sharpe_ratio": float(sharpe_ratio(daily_returns)),
        "sortino_ratio": float(sortino_ratio(daily_returns)),
        "calmar_ratio": float(calmar_ratio(equity_curve)),
        "max_drawdown": float(max_drawdown(equity_curve)),
        "drawdown_duration_days": float(drawdown_duration(drawdowns)),
        "ulcer_index": float(ulcer_index(drawdowns)),
        "downside_deviation": float(downside_deviation(daily_returns)),
        "skewness": distribution["skewness"],
        "kurtosis": distribution["kurtosis"],
        "win_rate": distribution["win_rate"],
    }


def _window_cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    days = (equity.index[-1] - equity.index[0]).days
    if days <= 0:
        return np.nan
    years = days / CALENDAR_DAYS
    if years <= 0 or equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return np.nan
    return (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0


def rolling_metrics(
    daily_returns: pd.Series,
    window_days: int,
) -> pd.DataFrame:
    daily_returns = _as_series(daily_returns).dropna()
    if len(daily_returns) < window_days:
        return pd.DataFrame(columns=["sharpe", "sortino", "calmar", "volatility"])

    records: list[dict[str, float]] = []
    index: list[pd.Timestamp] = []
    for start_idx in range(0, len(daily_returns) - window_days + 1):
        window = daily_returns.iloc[start_idx : start_idx + window_days]
        equity = equity_curve_from_returns(window)
        record = {
            "sharpe": sharpe_ratio(window),
            "sortino": sortino_ratio(window),
            "calmar": calmar_ratio(equity),
            "volatility": annualized_volatility(window),
        }
        records.append(record)
        index.append(window.index[-1])

    return pd.DataFrame(records, index=index)
