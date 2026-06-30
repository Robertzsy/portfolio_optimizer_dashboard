from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class PortfolioResult:
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe: float


def portfolio_metrics(
    weights: np.ndarray,
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    ret = float(np.dot(weights, expected_returns.to_numpy()))
    vol = float(np.sqrt(weights @ covariance.to_numpy() @ weights))
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else np.nan
    return ret, vol, sharpe


def _bounds(n_assets: int, max_weight: float) -> list[tuple[float, float]]:
    return [(0.0, max_weight) for _ in range(n_assets)]


def _weights_series(values: np.ndarray, index: pd.Index) -> pd.Series:
    clipped = np.clip(values, 0, 1)
    total = clipped.sum()
    if total == 0:
        clipped = np.repeat(1 / len(clipped), len(clipped))
    else:
        clipped = clipped / total
    return pd.Series(clipped, index=index, name="Weight")


def optimize_max_sharpe(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float,
    max_weight: float = 1.0,
) -> PortfolioResult:
    n_assets = len(expected_returns)
    initial = np.repeat(1 / n_assets, n_assets)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    def objective(weights: np.ndarray) -> float:
        _, _, sharpe = portfolio_metrics(weights, expected_returns, covariance, risk_free_rate)
        return -sharpe if np.isfinite(sharpe) else 1e6

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=_bounds(n_assets, max_weight),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    weights = _weights_series(result.x if result.success else initial, expected_returns.index)
    ret, vol, sharpe = portfolio_metrics(weights.to_numpy(), expected_returns, covariance, risk_free_rate)
    return PortfolioResult(weights, ret, vol, sharpe)


def optimize_min_variance(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
) -> PortfolioResult:
    n_assets = len(expected_returns)
    initial = np.repeat(1 / n_assets, n_assets)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    def objective(weights: np.ndarray) -> float:
        return float(weights @ covariance.to_numpy() @ weights)

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=_bounds(n_assets, max_weight),
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-10},
    )
    weights = _weights_series(result.x if result.success else initial, expected_returns.index)
    ret, vol, sharpe = portfolio_metrics(weights.to_numpy(), expected_returns, covariance, risk_free_rate)
    return PortfolioResult(weights, ret, vol, sharpe)


def efficient_frontier(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    max_weight: float = 1.0,
    points: int = 50,
) -> pd.DataFrame:
    n_assets = len(expected_returns)
    initial = np.repeat(1 / n_assets, n_assets)
    returns = expected_returns.to_numpy()
    min_ret = float(expected_returns.min())
    max_ret = float(expected_returns.max())
    targets = np.linspace(min_ret, max_ret, points)
    rows: list[dict[str, float]] = []

    for target in targets:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: np.dot(w, returns) - target},
        )
        result = minimize(
            lambda w: float(w @ covariance.to_numpy() @ w),
            initial,
            method="SLSQP",
            bounds=_bounds(n_assets, max_weight),
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if result.success:
            ret, vol, sharpe = portfolio_metrics(result.x, expected_returns, covariance, risk_free_rate)
            rows.append({"Return": ret, "Volatility": vol, "Sharpe": sharpe})

    return pd.DataFrame(rows).drop_duplicates()


def random_portfolios(
    expected_returns: pd.Series,
    covariance: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n: int = 2500,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        weights = rng.dirichlet(np.ones(len(expected_returns)))
        ret, vol, sharpe = portfolio_metrics(weights, expected_returns, covariance, risk_free_rate)
        rows.append({"Return": ret, "Volatility": vol, "Sharpe": sharpe})
    return pd.DataFrame(rows)

