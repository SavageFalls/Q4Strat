"""Performance and risk analytics for seasonal strategy research."""

from __future__ import annotations

import numpy as np
import pandas as pd


TRADING_DAYS = 252


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0 or np.isnan(denominator):
        return np.nan
    return numerator / denominator


def drawdown_series(equity: pd.Series) -> pd.Series:
    running_max = equity.cummax()
    return equity / running_max - 1.0


def max_drawdown(drawdown: pd.Series) -> float:
    return float(drawdown.min()) if len(drawdown) else np.nan


def drawdown_duration(drawdown: pd.Series) -> int:
    """Longest streak in drawdown (days below prior peak)."""
    underwater = drawdown < 0
    groups = (underwater != underwater.shift()).cumsum()
    durations = underwater.groupby(groups).sum()
    return int(durations.max()) if len(durations) else 0


def ulcer_index(drawdown: pd.Series) -> float:
    dd_pct = drawdown * 100.0
    return float(np.sqrt(np.mean(np.square(dd_pct)))) if len(dd_pct) else np.nan


def downside_deviation(returns: pd.Series, mar: float = 0.0, annualize: bool = True) -> float:
    downside = np.minimum(returns - mar, 0.0)
    dd = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else np.nan
    if annualize and not np.isnan(dd):
        dd *= np.sqrt(TRADING_DAYS)
    return dd


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return np.nan
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    if years <= 0:
        return np.nan
    return float(equity.iloc[-1] ** (1.0 / years) - 1.0)


def skewness(returns: pd.Series) -> float:
    return float(returns.skew()) if len(returns) else np.nan


def kurtosis(returns: pd.Series) -> float:
    return float(returns.kurt()) if len(returns) else np.nan


def win_rate(returns: pd.Series) -> float:
    if len(returns) == 0:
        return np.nan
    return float((returns > 0).mean())


def compute_performance_metrics(returns: pd.Series, equity: pd.Series) -> dict[str, float]:
    rf = 0.0
    ann_ret = cagr(equity)
    total_ret = float(equity.iloc[-1] - 1.0) if len(equity) else np.nan
    ann_vol = annualized_volatility(returns)
    dd = drawdown_series(equity)
    mdd = max_drawdown(dd)
    downside = downside_deviation(returns, mar=0.0, annualize=True)

    sharpe = _safe_div(ann_ret - rf, ann_vol)
    sortino = _safe_div(ann_ret - rf, downside)
    calmar = _safe_div(ann_ret, abs(mdd))

    return {
        "cagr": ann_ret,
        "total_return": total_ret,
        "annualized_volatility": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": mdd,
        "drawdown_duration": drawdown_duration(dd),
        "ulcer_index": ulcer_index(dd),
        "downside_deviation": downside,
        "skewness": skewness(returns),
        "kurtosis": kurtosis(returns),
        "win_rate": win_rate(returns),
    }


def rolling_metrics(
    returns: pd.Series,
    equity: pd.Series,
    window_years: int = 5,
) -> pd.DataFrame:
    window = max(2, int(window_years * TRADING_DAYS))
    df = pd.DataFrame(index=returns.index)

    roll_ann_ret = (1.0 + returns).rolling(window).apply(
        lambda x: x.prod() ** (TRADING_DAYS / len(x)) - 1.0,
        raw=False,
    )

    roll_vol = returns.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)

    roll_downside = returns.rolling(window).apply(
        lambda x: np.sqrt(np.mean(np.square(np.minimum(x, 0.0)))) * np.sqrt(TRADING_DAYS),
        raw=False,
    )

    roll_sharpe = (roll_ann_ret) / roll_vol
    roll_sortino = (roll_ann_ret) / roll_downside

    roll_mdd = equity.rolling(window).apply(
        lambda x: (pd.Series(x) / pd.Series(x).cummax() - 1.0).min(),
        raw=False,
    )
    roll_calmar = roll_ann_ret / roll_mdd.abs()

    df["rolling_sharpe"] = roll_sharpe
    df["rolling_sortino"] = roll_sortino
    df["rolling_calmar"] = roll_calmar
    df["rolling_volatility"] = roll_vol

    return df
