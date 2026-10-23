"""Non-parametric KNN conditional-moment estimator (Algorithm 2, Sec. 7.2/8).

Context7 check: `sklearn.neighbors.NearestNeighbors` (named in the
implementation notes) is used for neighbor retrieval, and
`sklearn.covariance.LedoitWolf` (also named) is used for shrinkage covariance
estimation on the neighbor return sample -- both are direct, documented
scikit-learn estimators, used exactly as named rather than reimplemented.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


def knn_conditional_moments(
    x_train: np.ndarray, r_train: np.ndarray, x_query: np.ndarray, n_neighbors: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve the K nearest neighbors of `x_query` in `x_train` (expanding
    history of feature vectors) and estimate:
      mu_t    = (1/K) * sum_{s in N_t} r_s                      (Sec. 7.2)
      Sigma_t = LedoitWolf shrinkage covariance of {r_s}_{s in N_t}
    """
    n_neighbors = min(n_neighbors, x_train.shape[0])
    nn = NearestNeighbors(n_neighbors=n_neighbors)
    nn.fit(x_train)
    _, indices = nn.kneighbors(x_query.reshape(1, -1))
    neighbor_returns = r_train[indices[0]]

    mu = neighbor_returns.mean(axis=0)
    if neighbor_returns.shape[0] > 1:
        lw = LedoitWolf()
        lw.fit(neighbor_returns)
        sigma = lw.covariance_
    else:
        sigma = np.eye(neighbor_returns.shape[1]) * 1e-6
    return mu, sigma
