from __future__ import annotations

import numpy as np
import pandas as pd


def apply_absolute_views(
    prior_returns: pd.Series,
    covariance: pd.DataFrame,
    views: dict[str, float],
    tau: float = 0.05,
    confidence: float = 0.6,
) -> pd.Series:
    """Blend historical expected returns with absolute investor views.

    The implementation is intentionally compact for dashboard use: each view says
    "ticker X should have annual return Y". Confidence controls how strongly the
    posterior is pulled toward those views.
    """
    clean_views = {asset: value for asset, value in views.items() if asset in prior_returns.index}
    if not clean_views:
        return prior_returns

    assets = list(prior_returns.index)
    p = np.zeros((len(clean_views), len(assets)))
    q = np.zeros(len(clean_views))
    for row, (asset, value) in enumerate(clean_views.items()):
        p[row, assets.index(asset)] = 1
        q[row] = value

    sigma = covariance.loc[assets, assets].to_numpy()
    pi = prior_returns.loc[assets].to_numpy()
    tau_sigma = tau * sigma
    view_variance = np.diag(np.diag(p @ tau_sigma @ p.T) / max(confidence, 1e-6))

    middle = np.linalg.inv(p @ tau_sigma @ p.T + view_variance)
    posterior = pi + tau_sigma @ p.T @ middle @ (q - p @ pi)
    return pd.Series(posterior, index=assets, name="BL Expected Return")

