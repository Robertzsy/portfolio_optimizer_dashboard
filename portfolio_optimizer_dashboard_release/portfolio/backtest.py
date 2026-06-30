from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .data import TRADING_DAYS, annualized_mean_cov
from .optimization import optimize_max_sharpe, optimize_min_variance


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.DataFrame
    weights: pd.DataFrame
    metrics: pd.DataFrame


def performance_metrics(series: pd.Series, risk_free_rate: float = 0.0) -> dict[str, float]:
    returns = series.pct_change().dropna()
    if returns.empty:
        return {"CAGR": np.nan, "Volatility": np.nan, "Sharpe": np.nan, "Max Drawdown": np.nan}

    years = len(returns) / TRADING_DAYS
    cagr = float(series.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = float((returns.mean() * TRADING_DAYS - risk_free_rate) / vol) if vol > 0 else np.nan
    drawdown = series / series.cummax() - 1
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Max Drawdown": float(drawdown.min()),
    }


def _last_trading_days(returns: pd.DataFrame, frequency: str) -> pd.DatetimeIndex:
    dates = returns.groupby(pd.Grouper(freq=frequency)).tail(1).index
    return pd.DatetimeIndex(dates).unique()


def rolling_backtest(
    returns: pd.DataFrame,
    strategy: str,
    risk_free_rate: float = 0.0,
    lookback_days: int = 252,
    rebalance_frequency: str = "ME",
    max_weight: float = 1.0,
    benchmark_returns: pd.Series | None = None,
    view_adjuster: Callable[[pd.Series, pd.DataFrame], pd.Series] | None = None,
) -> BacktestResult:
    if len(returns) <= lookback_days + 5:
        raise ValueError("Not enough observations for the selected rolling window.")

    rebalance_dates = [date for date in _last_trading_days(returns, rebalance_frequency) if returns.index.get_loc(date) >= lookback_days]
    if not rebalance_dates:
        raise ValueError("No rebalance dates found. Try a shorter lookback window.")

    portfolio_returns = pd.Series(0.0, index=returns.index, name="Optimized")
    weights_by_date: dict[pd.Timestamp, pd.Series] = {}

    for i, date in enumerate(rebalance_dates):
        start_loc = returns.index.get_loc(date) - lookback_days
        train = returns.iloc[start_loc : returns.index.get_loc(date)]
        expected, covariance = annualized_mean_cov(train)
        if view_adjuster is not None:
            expected = view_adjuster(expected, covariance)

        if strategy == "Minimum Variance":
            result = optimize_min_variance(expected, covariance, risk_free_rate, max_weight)
        else:
            result = optimize_max_sharpe(expected, covariance, risk_free_rate, max_weight)

        weights_by_date[pd.Timestamp(date)] = result.weights
        start = returns.index.get_loc(date) + 1
        end = returns.index.get_loc(rebalance_dates[i + 1]) + 1 if i + 1 < len(rebalance_dates) else len(returns)
        if start < end:
            portfolio_returns.iloc[start:end] = returns.iloc[start:end].dot(result.weights)

    active = portfolio_returns.ne(0)
    first_active = active.idxmax()
    portfolio_returns = portfolio_returns.loc[first_active:]
    equal_weight = returns.loc[portfolio_returns.index].mean(axis=1).rename("Equal Weight")

    equity = pd.DataFrame(
        {
            "Optimized": (1 + portfolio_returns).cumprod(),
            "Equal Weight": (1 + equal_weight).cumprod(),
        }
    )

    if benchmark_returns is not None and not benchmark_returns.empty:
        benchmark = benchmark_returns.reindex(portfolio_returns.index).fillna(0)
        equity["Benchmark"] = (1 + benchmark).cumprod()

    metrics = pd.DataFrame({col: performance_metrics(equity[col], risk_free_rate) for col in equity.columns}).T
    weights = pd.DataFrame(weights_by_date).T.fillna(0)
    return BacktestResult(equity=equity, weights=weights, metrics=metrics)

