"""Persistent regime-template tracking (Sec. 5.4, "Template Tracking").

Each day's fitted K_t HMM components are mapped to a fixed bank of G
persistent templates by solving the assignment problem that minimizes total
closed-form 2-Wasserstein distance between components and templates (one
template per component, no template reused twice on the same day). Matched
templates are then updated by exponential smoothing with rate eta; unmatched
templates persist unchanged and receive probability 0 for the day.

Context7 check: `scipy.optimize.linear_sum_assignment` (Hungarian algorithm)
is the standard library solver for this bipartite min-cost assignment; no
paper-specific optimizer is implied, so the library routine is used directly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.wasserstein import gaussian_w2_squared


@dataclass
class TemplateBank:
    means: np.ndarray           # (G, D)
    covars: np.ndarray          # (G, D, D)
    smoothing_rate: float
    n_templates: int = field(init=False)

    def __post_init__(self):
        self.n_templates = self.means.shape[0]

    @classmethod
    def initialize(
        cls, calibration_X: np.ndarray, n_templates: int, smoothing_rate: float, random_state: int = 0
    ) -> "TemplateBank":
        """Initialize templates by fitting a G-component Gaussian mixture on
        the initial calibration window (Algorithm 1, line 3: "initialize
        templates using an initial calibration window")."""
        from sklearn.mixture import GaussianMixture

        gmm = GaussianMixture(
            n_components=n_templates,
            covariance_type="full",
            random_state=random_state,
            reg_covar=1e-6,
        )
        gmm.fit(calibration_X)
        return cls(means=gmm.means_.copy(), covars=gmm.covariances_.copy(), smoothing_rate=smoothing_rate)

    def map_and_update(
        self, component_probs: np.ndarray, component_means: np.ndarray, component_covars: np.ndarray
    ) -> np.ndarray:
        """Map today's K components to templates via min-cost 2-Wasserstein
        assignment, update matched templates by EMA, and return the
        per-template probability vector p_{t,g} (length G, zeros where
        unmatched)."""
        k = component_probs.shape[0]
        g = self.n_templates
        cost = np.zeros((k, g))
        for i in range(k):
            for j in range(g):
                cost[i, j] = gaussian_w2_squared(
                    component_means[i], component_covars[i], self.means[j], self.covars[j]
                )
        row_ind, col_ind = linear_sum_assignment(cost)

        template_probs = np.zeros(g)
        eta = self.smoothing_rate
        for i, j in zip(row_ind, col_ind):
            template_probs[j] = component_probs[i]
            self.means[j] = (1 - eta) * self.means[j] + eta * component_means[i]
            self.covars[j] = (1 - eta) * self.covars[j] + eta * component_covars[i]
        return template_probs

    def aggregate_moments(self, template_probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """mu_t = sum_g p_{t,g} mu_g ; Sigma_t = sum_g p_{t,g} Sigma_g (Sec. 5.5)."""
        mu_t = np.tensordot(template_probs, self.means, axes=(0, 0))
        sigma_t = np.tensordot(template_probs, self.covars, axes=(0, 0))
        return mu_t, sigma_t
