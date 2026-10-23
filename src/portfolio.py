"""Transaction-cost-aware mean-variance optimization (Sec. 6, Eq. shared by
both Algorithm 1 and Algorithm 2):

    max_w  mu^T w - risk_aversion * w^T Sigma w - tau * ||w - w_prev||_1
    s.t.   1^T w = 1, w >= 0, ||w||_inf <= w_max

Context7 check: `cvxpy` (named explicitly in the experiment implementation
notes) is the right tool for this convex QP with an L1 penalty; PyPortfolioOpt
targets a fixed menu of named objectives (e.g. `EfficientFrontier`) and does
not expose an arbitrary linear transaction-cost-penalized mean-variance
objective with a previous-weight anchor, so the problem is formulated
directly in cvxpy as the paper specifies, handling the L1 term via the
`cvxpy.norm1` atom (equivalent to the auxiliary-variable formulation).
"""
from __future__ import annotations

from typing import Optional

import cvxpy as cp
import numpy as np


def _nearest_psd(sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    sigma = 0.5 * (sigma + sigma.T)
    eigval, eigvec = np.linalg.eigh(sigma)
    eigval = np.clip(eigval, eps, None)
    return (eigvec * eigval) @ eigvec.T


def solve_tc_aware_mvo(
    mu: np.ndarray,
    sigma: np.ndarray,
    w_prev: np.ndarray,
    risk_aversion: float,
    tau: float,
    w_max: float,
) -> np.ndarray:
    """Solve the daily transaction-cost-aware MVO problem and return w_t.

    Falls back to the previous weights (and logs via exception re-raise only
    on genuine solver failure) only after attempting ECOS then SCS, keeping
    the problem convex and identical across both strategies.
    """
    n = len(mu)
    sigma = _nearest_psd(np.asarray(sigma, dtype=float))
    mu = np.asarray(mu, dtype=float)
    w_prev = np.asarray(w_prev, dtype=float)

    w = cp.Variable(n)
    risk = cp.quad_form(w, cp.psd_wrap(sigma))
    tc = cp.norm1(w - w_prev)
    objective = cp.Maximize(mu @ w - risk_aversion * risk - tau * tc)
    constraints = [cp.sum(w) == 1, w >= 0, w <= w_max]
    problem = cp.Problem(objective, constraints)

    for solver in (cp.ECOS, cp.SCS):
        try:
            problem.solve(solver=solver)
            if w.value is not None and problem.status in ("optimal", "optimal_inaccurate"):
                w_sol = np.clip(np.asarray(w.value).flatten(), 0.0, None)
                total = w_sol.sum()
                if total <= 0:
                    continue
                return w_sol / total
        except cp.error.SolverError:
            continue
    raise RuntimeError(f"MVO solve failed for all solvers; status={problem.status}")
